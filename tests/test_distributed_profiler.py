"""Tests for src.profiler.distributed_profiler."""

import time
from unittest.mock import patch

import pytest

from src.profiler.distributed_profiler import (
    AllReduceEvent,
    GradSyncEvent,
    DistributedProfileResult,
    DistributedProfiler,
)


# ---------------------------------------------------------------------------
# AllReduceEvent
# ---------------------------------------------------------------------------

class TestAllReduceEvent:
    def test_creation(self):
        e = AllReduceEvent(
            timestamp=1000.0, duration_us=500.0,
            data_size_bytes=1024, step_index=0,
        )
        assert e.duration_us == 500.0
        assert e.data_size_bytes == 1024

    def test_to_dict_from_dict_roundtrip(self):
        e = AllReduceEvent(
            timestamp=2000.0, duration_us=750.0,
            data_size_bytes=4096, step_index=5,
        )
        d = e.to_dict()
        restored = AllReduceEvent.from_dict(d)
        assert restored.timestamp == e.timestamp
        assert restored.duration_us == e.duration_us
        assert restored.step_index == e.step_index

    def test_from_dict_defaults(self):
        e = AllReduceEvent.from_dict({})
        assert e.timestamp == 0.0
        assert e.data_size_bytes == 0


# ---------------------------------------------------------------------------
# GradSyncEvent
# ---------------------------------------------------------------------------

class TestGradSyncEvent:
    def test_creation(self):
        e = GradSyncEvent(
            step_index=0, compute_time_us=8000.0,
            sync_time_us=2000.0, total_step_time_us=10000.0,
            timestamp=1000.0,
        )
        assert e.compute_time_us == 8000.0
        assert e.sync_time_us == 2000.0

    def test_to_dict_from_dict_roundtrip(self):
        e = GradSyncEvent(
            step_index=3, compute_time_us=5000.0,
            sync_time_us=1000.0, total_step_time_us=6000.0,
            timestamp=2000.0,
        )
        d = e.to_dict()
        restored = GradSyncEvent.from_dict(d)
        assert restored.step_index == e.step_index
        assert restored.compute_time_us == e.compute_time_us


# ---------------------------------------------------------------------------
# DistributedProfileResult
# ---------------------------------------------------------------------------

class TestDistributedProfileResult:
    def test_empty_result(self):
        result = DistributedProfileResult(
            allreduce_events=[], grad_sync_events=[],
            total_allreduce_time_us=0.0, avg_allreduce_time_us=0.0,
            total_grad_sync_time_us=0.0, avg_grad_sync_time_us=0.0,
            total_compute_time_us=0.0, comm_compute_ratio=0.0,
            gradient_sync_overhead_pct=0.0, world_size=0,
            num_nodes=0, recommendations=[],
        )
        assert result.world_size == 0
        assert result.comm_compute_ratio == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        ar_event = AllReduceEvent(
            timestamp=1.0, duration_us=500.0, data_size_bytes=1024, step_index=0,
        )
        gs_event = GradSyncEvent(
            step_index=0, compute_time_us=8000.0, sync_time_us=2000.0,
            total_step_time_us=10000.0, timestamp=1.0,
        )
        result = DistributedProfileResult(
            allreduce_events=[ar_event], grad_sync_events=[gs_event],
            total_allreduce_time_us=500.0, avg_allreduce_time_us=500.0,
            total_grad_sync_time_us=2000.0, avg_grad_sync_time_us=2000.0,
            total_compute_time_us=8000.0, comm_compute_ratio=0.25,
            gradient_sync_overhead_pct=20.0, world_size=4,
            num_nodes=2, recommendations=["Use gradient bucketing."],
        )
        d = result.to_dict()
        restored = DistributedProfileResult.from_dict(d)
        assert restored.world_size == 4
        assert len(restored.allreduce_events) == 1
        assert len(restored.grad_sync_events) == 1
        assert restored.recommendations == ["Use gradient bucketing."]

    def test_from_dict_defaults(self):
        result = DistributedProfileResult.from_dict({})
        assert result.world_size == 0
        assert result.allreduce_events == []


# ---------------------------------------------------------------------------
# DistributedProfiler — lifecycle
# ---------------------------------------------------------------------------

class TestDistributedProfilerLifecycle:
    def test_start_stop(self):
        profiler = DistributedProfiler()
        profiler.start()
        result = profiler.stop()
        assert isinstance(result, DistributedProfileResult)
        assert result.comm_compute_ratio == 0.0

    def test_stop_without_start_returns_empty(self):
        profiler = DistributedProfiler()
        result = profiler.stop()
        assert isinstance(result, DistributedProfileResult)
        assert result.world_size == 0

    def test_duplicate_start_ignored(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.start()  # should not crash
        profiler.stop()

    def test_record_allreduce(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_allreduce(duration_us=500.0, data_size_bytes=1024, step_index=0)
        profiler.record_allreduce(duration_us=600.0, data_size_bytes=2048, step_index=1)
        result = profiler.stop()
        assert len(result.allreduce_events) == 2
        assert result.total_allreduce_time_us == pytest.approx(1100.0)
        assert result.avg_allreduce_time_us == pytest.approx(550.0)

    def test_record_allreduce_ignored_when_not_active(self):
        profiler = DistributedProfiler()
        profiler.record_allreduce(duration_us=500.0)
        profiler.start()
        result = profiler.stop()
        assert len(result.allreduce_events) == 0

    def test_record_step(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_step(step_index=0, compute_time_us=8000.0, sync_time_us=2000.0)
        profiler.record_step(step_index=1, compute_time_us=7000.0, sync_time_us=3000.0)
        result = profiler.stop()
        assert len(result.grad_sync_events) == 2
        assert result.total_compute_time_us == pytest.approx(15000.0)
        assert result.total_grad_sync_time_us == pytest.approx(5000.0)

    def test_record_step_ignored_when_not_active(self):
        profiler = DistributedProfiler()
        profiler.record_step(step_index=0, compute_time_us=8000.0, sync_time_us=2000.0)
        profiler.start()
        result = profiler.stop()
        assert len(result.grad_sync_events) == 0


# ---------------------------------------------------------------------------
# DistributedProfiler — metrics
# ---------------------------------------------------------------------------

class TestDistributedProfilerMetrics:
    def test_comm_compute_ratio(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_step(0, compute_time_us=8000.0, sync_time_us=2000.0)
        profiler.record_step(1, compute_time_us=8000.0, sync_time_us=2000.0)
        ratio = profiler.get_comm_compute_ratio()
        assert ratio == pytest.approx(0.25)  # 4000 / 16000
        profiler.stop()

    def test_comm_compute_ratio_no_data(self):
        profiler = DistributedProfiler()
        assert profiler.get_comm_compute_ratio() == 0.0

    def test_gradient_sync_overhead(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_step(0, compute_time_us=8000.0, sync_time_us=2000.0)
        overhead = profiler.get_gradient_sync_overhead()
        assert overhead == pytest.approx(20.0)  # 2000/10000 * 100
        profiler.stop()

    def test_gradient_sync_overhead_no_data(self):
        profiler = DistributedProfiler()
        assert profiler.get_gradient_sync_overhead() == 0.0

    def test_comm_compute_ratio_in_result(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_step(0, compute_time_us=5000.0, sync_time_us=5000.0)
        result = profiler.stop()
        assert result.comm_compute_ratio == pytest.approx(1.0)
        assert result.gradient_sync_overhead_pct == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# DistributedProfiler — recommendations
# ---------------------------------------------------------------------------

class TestDistributedProfilerRecommendations:
    def test_high_comm_ratio_recommendation(self):
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.8, overhead_pct=20.0,
            allreduce_events=[], world_size=4,
        )
        assert any("ratio" in r.lower() for r in recs)

    def test_high_overhead_recommendation(self):
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.1, overhead_pct=35.0,
            allreduce_events=[], world_size=4,
        )
        assert any("bucketing" in r.lower() for r in recs)

    def test_very_high_overhead_fsdp_recommendation(self):
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.1, overhead_pct=55.0,
            allreduce_events=[], world_size=4,
        )
        assert any("fsdp" in r.lower() for r in recs)

    def test_small_message_recommendation(self):
        events = [
            AllReduceEvent(timestamp=1.0, duration_us=100.0, data_size_bytes=500, step_index=0),
            AllReduceEvent(timestamp=2.0, duration_us=100.0, data_size_bytes=600, step_index=1),
        ]
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.1, overhead_pct=10.0,
            allreduce_events=events, world_size=4,
        )
        assert any("bucket" in r.lower() for r in recs)

    def test_large_world_size_recommendation(self):
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.1, overhead_pct=25.0,
            allreduce_events=[], world_size=16,
        )
        assert any("hierarchical" in r.lower() for r in recs)

    def test_no_recommendations_when_healthy(self):
        recs = DistributedProfiler._generate_recommendations(
            comm_compute_ratio=0.1, overhead_pct=5.0,
            allreduce_events=[], world_size=4,
        )
        assert len(recs) == 0

    def test_auto_increment_step_counter(self):
        profiler = DistributedProfiler()
        profiler.start()
        profiler.record_allreduce(duration_us=100.0)
        profiler.record_step(step_index=0, compute_time_us=5000.0, sync_time_us=1000.0)
        profiler.record_allreduce(duration_us=200.0)
        result = profiler.stop()
        # First allreduce uses step_counter 0, second uses step_counter 1
        assert result.allreduce_events[0].step_index == 0
        assert result.allreduce_events[1].step_index == 1
