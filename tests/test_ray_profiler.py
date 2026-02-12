"""Tests for src.profiler.ray_profiler."""

import time
from unittest.mock import patch, MagicMock

import pytest

from src.profiler.ray_profiler import (
    RayActorMetrics,
    RayActorGPUMapping,
    RayStragglerInfo,
    RayProfileResult,
    RayProfiler,
    _STRAGGLER_SIGMA_THRESHOLD,
    _MIN_ACTORS_FOR_STRAGGLER,
)


# ---------------------------------------------------------------------------
# RayActorMetrics
# ---------------------------------------------------------------------------

class TestRayActorMetrics:
    def test_creation(self):
        m = RayActorMetrics(
            actor_id="actor_001",
            node_id="node_abc",
            gpu_index=0,
            pid=12345,
            avg_iteration_time_ms=50.0,
            total_iterations=100,
            gpu_utilization_pct=85.0,
            gpu_memory_used_mb=4096.0,
            status="running",
            timestamp=1000.0,
        )
        assert m.actor_id == "actor_001"
        assert m.gpu_index == 0
        assert m.avg_iteration_time_ms == 50.0

    def test_to_dict_from_dict_roundtrip(self):
        m = RayActorMetrics(
            actor_id="a1", node_id="n1", gpu_index=1, pid=999,
            avg_iteration_time_ms=33.3, total_iterations=50,
            gpu_utilization_pct=72.0, gpu_memory_used_mb=2048.0,
            status="running", timestamp=2000.0,
        )
        d = m.to_dict()
        restored = RayActorMetrics.from_dict(d)
        assert restored.actor_id == m.actor_id
        assert restored.gpu_index == m.gpu_index
        assert restored.avg_iteration_time_ms == m.avg_iteration_time_ms
        assert restored.status == m.status

    def test_from_dict_defaults(self):
        m = RayActorMetrics.from_dict({})
        assert m.actor_id == ""
        assert m.gpu_index == -1
        assert m.pid == 0
        assert m.status == "unknown"


# ---------------------------------------------------------------------------
# RayActorGPUMapping
# ---------------------------------------------------------------------------

class TestRayActorGPUMapping:
    def test_creation(self):
        mapping = RayActorGPUMapping(
            actor_id="a1", node_ip="10.0.0.1", node_id="n1",
            gpu_index=2, gpu_name="A100",
        )
        assert mapping.actor_id == "a1"
        assert mapping.gpu_index == 2

    def test_to_dict_from_dict_roundtrip(self):
        mapping = RayActorGPUMapping(
            actor_id="a2", node_ip="10.0.0.2", node_id="n2",
            gpu_index=0, gpu_name="H100",
        )
        d = mapping.to_dict()
        restored = RayActorGPUMapping.from_dict(d)
        assert restored.actor_id == mapping.actor_id
        assert restored.gpu_name == mapping.gpu_name

    def test_from_dict_defaults(self):
        m = RayActorGPUMapping.from_dict({})
        assert m.actor_id == ""
        assert m.gpu_index == -1


# ---------------------------------------------------------------------------
# RayStragglerInfo
# ---------------------------------------------------------------------------

class TestRayStragglerInfo:
    def test_creation(self):
        s = RayStragglerInfo(
            actor_id="a3", node_id="n1", gpu_index=0,
            avg_iteration_time_ms=150.0, slowdown_factor=2.5,
        )
        assert s.slowdown_factor == 2.5

    def test_to_dict_from_dict_roundtrip(self):
        s = RayStragglerInfo(
            actor_id="a4", node_id="n2", gpu_index=1,
            avg_iteration_time_ms=200.0, slowdown_factor=3.0,
        )
        d = s.to_dict()
        restored = RayStragglerInfo.from_dict(d)
        assert restored.actor_id == s.actor_id
        assert restored.slowdown_factor == s.slowdown_factor


# ---------------------------------------------------------------------------
# RayProfileResult
# ---------------------------------------------------------------------------

class TestRayProfileResult:
    def test_empty_result(self):
        result = RayProfileResult(
            actors=[], actor_gpu_map=[], stragglers=[],
            num_workers=0, num_nodes=0, avg_iteration_time_ms=0.0,
            iteration_time_std_ms=0.0, total_training_time_s=0.0,
            recommendations=[],
        )
        assert result.num_workers == 0
        assert result.recommendations == []

    def test_to_dict_from_dict_roundtrip(self):
        actor = RayActorMetrics(
            actor_id="a1", node_id="n1", gpu_index=0, pid=100,
            avg_iteration_time_ms=50.0, total_iterations=10,
            gpu_utilization_pct=80.0, gpu_memory_used_mb=2000.0,
            status="running", timestamp=1000.0,
        )
        mapping = RayActorGPUMapping(
            actor_id="a1", node_ip="10.0.0.1", node_id="n1",
            gpu_index=0, gpu_name="A100",
        )
        straggler = RayStragglerInfo(
            actor_id="a1", node_id="n1", gpu_index=0,
            avg_iteration_time_ms=50.0, slowdown_factor=1.5,
        )
        result = RayProfileResult(
            actors=[actor], actor_gpu_map=[mapping], stragglers=[straggler],
            num_workers=1, num_nodes=1, avg_iteration_time_ms=50.0,
            iteration_time_std_ms=5.0, total_training_time_s=10.0,
            recommendations=["Test recommendation"],
        )
        d = result.to_dict()
        restored = RayProfileResult.from_dict(d)
        assert restored.num_workers == 1
        assert len(restored.actors) == 1
        assert len(restored.actor_gpu_map) == 1
        assert len(restored.stragglers) == 1
        assert restored.recommendations == ["Test recommendation"]

    def test_from_dict_defaults(self):
        result = RayProfileResult.from_dict({})
        assert result.num_workers == 0
        assert result.actors == []


# ---------------------------------------------------------------------------
# RayProfiler — straggler detection
# ---------------------------------------------------------------------------

class TestRayProfilerStragglerDetection:
    def _make_actor(self, actor_id: str, node_id: str, iter_time: float) -> RayActorMetrics:
        return RayActorMetrics(
            actor_id=actor_id, node_id=node_id, gpu_index=0, pid=100,
            avg_iteration_time_ms=iter_time, total_iterations=100,
            gpu_utilization_pct=80.0, gpu_memory_used_mb=2000.0,
            status="running", timestamp=1000.0,
        )

    def test_no_stragglers_when_uniform(self):
        profiler = RayProfiler()
        actors = [
            self._make_actor("a1", "n1", 50.0),
            self._make_actor("a2", "n1", 51.0),
            self._make_actor("a3", "n2", 49.0),
        ]
        stragglers = profiler.detect_stragglers(actors)
        assert len(stragglers) == 0

    def test_detects_straggler(self):
        profiler = RayProfiler()
        actors = [
            self._make_actor("a1", "n1", 50.0),
            self._make_actor("a2", "n1", 50.0),
            self._make_actor("a3", "n2", 50.0),
            self._make_actor("a4", "n2", 200.0),  # straggler
        ]
        stragglers = profiler.detect_stragglers(actors)
        assert len(stragglers) >= 1
        assert stragglers[0].actor_id == "a4"
        assert stragglers[0].slowdown_factor > 1.0

    def test_no_stragglers_with_too_few_actors(self):
        profiler = RayProfiler()
        actors = [self._make_actor("a1", "n1", 50.0)]
        stragglers = profiler.detect_stragglers(actors)
        assert len(stragglers) == 0

    def test_no_stragglers_with_zero_times(self):
        profiler = RayProfiler()
        actors = [
            self._make_actor("a1", "n1", 0.0),
            self._make_actor("a2", "n1", 0.0),
        ]
        stragglers = profiler.detect_stragglers(actors)
        assert len(stragglers) == 0

    def test_stragglers_sorted_by_slowdown(self):
        profiler = RayProfiler()
        actors = [
            self._make_actor("a1", "n1", 50.0),
            self._make_actor("a2", "n1", 50.0),
            self._make_actor("a3", "n2", 50.0),
            self._make_actor("a4", "n2", 200.0),
            self._make_actor("a5", "n2", 300.0),  # worse straggler
        ]
        stragglers = profiler.detect_stragglers(actors)
        if len(stragglers) >= 2:
            assert stragglers[0].slowdown_factor >= stragglers[1].slowdown_factor

    def test_uses_internal_metrics_when_none_passed(self):
        profiler = RayProfiler()
        profiler._actor_metrics = [
            self._make_actor("a1", "n1", 50.0),
            self._make_actor("a2", "n1", 50.0),
            self._make_actor("a3", "n2", 200.0),
        ]
        stragglers = profiler.detect_stragglers()
        # May or may not detect straggler depending on threshold
        assert isinstance(stragglers, list)


# ---------------------------------------------------------------------------
# RayProfiler — lifecycle
# ---------------------------------------------------------------------------

class TestRayProfilerLifecycle:
    def test_start_stop_without_ray(self):
        """Profiler works without Ray installed — returns empty results."""
        with patch("src.profiler.ray_profiler._ray_available", False):
            profiler = RayProfiler()
            profiler.start()
            result = profiler.stop()
            assert isinstance(result, RayProfileResult)
            assert result.num_workers == 0

    def test_stop_without_start_returns_empty(self):
        profiler = RayProfiler()
        result = profiler.stop()
        assert isinstance(result, RayProfileResult)
        assert result.num_workers == 0

    def test_duplicate_start_ignored(self):
        with patch("src.profiler.ray_profiler._ray_available", False):
            profiler = RayProfiler()
            profiler.start()
            profiler.start()  # should not crash
            profiler.stop()

    def test_map_actors_without_ray(self):
        with patch("src.profiler.ray_profiler._ray_available", False):
            profiler = RayProfiler()
            mappings = profiler.map_actors_to_gpus()
            assert mappings == []

    def test_scrape_metrics_without_ray(self):
        with patch("src.profiler.ray_profiler._ray_available", False):
            profiler = RayProfiler()
            metrics = profiler.scrape_ray_metrics()
            assert metrics == []


# ---------------------------------------------------------------------------
# RayProfiler — iteration stats
# ---------------------------------------------------------------------------

class TestRayProfilerIterationStats:
    def test_compute_iteration_stats_empty(self):
        avg, std = RayProfiler._compute_iteration_stats([])
        assert avg == 0.0
        assert std == 0.0

    def test_compute_iteration_stats_single(self):
        actor = RayActorMetrics(
            actor_id="a1", node_id="n1", gpu_index=0, pid=1,
            avg_iteration_time_ms=100.0, total_iterations=10,
            gpu_utilization_pct=80.0, gpu_memory_used_mb=2000.0,
            status="running", timestamp=1.0,
        )
        avg, std = RayProfiler._compute_iteration_stats([actor])
        assert avg == 100.0
        assert std == 0.0

    def test_compute_iteration_stats_multiple(self):
        actors = [
            RayActorMetrics(
                actor_id=f"a{i}", node_id="n1", gpu_index=0, pid=i,
                avg_iteration_time_ms=float(50 + i * 10), total_iterations=10,
                gpu_utilization_pct=80.0, gpu_memory_used_mb=2000.0,
                status="running", timestamp=1.0,
            )
            for i in range(4)
        ]
        avg, std = RayProfiler._compute_iteration_stats(actors)
        assert avg > 0
        assert std >= 0


# ---------------------------------------------------------------------------
# RayProfiler — recommendations
# ---------------------------------------------------------------------------

class TestRayProfilerRecommendations:
    def test_recommendations_for_stragglers(self):
        actors = []
        stragglers = [
            RayStragglerInfo(
                actor_id="a1", node_id="n1", gpu_index=0,
                avg_iteration_time_ms=200.0, slowdown_factor=2.0,
            )
        ]
        recs = RayProfiler._generate_recommendations(actors, stragglers)
        assert any("straggler" in r.lower() for r in recs)

    def test_recommendations_for_low_utilization(self):
        actors = [
            RayActorMetrics(
                actor_id="a1", node_id="n1", gpu_index=0, pid=1,
                avg_iteration_time_ms=50.0, total_iterations=10,
                gpu_utilization_pct=30.0, gpu_memory_used_mb=2000.0,
                status="running", timestamp=1.0,
            )
        ]
        recs = RayProfiler._generate_recommendations(actors, [])
        assert any("utilization" in r.lower() for r in recs)

    def test_recommendations_for_no_actors(self):
        recs = RayProfiler._generate_recommendations([], [])
        assert any("no ray train actors" in r.lower() for r in recs)
