"""Tests for src.chamber.ray_integration."""

from unittest.mock import patch

import pytest

from src.chamber.ray_integration import (
    WorkerProfile,
    CommunicationBottleneck,
    RayTrainProfileResult,
    RayTrainIntegration,
    _IMBALANCE_RATIO_THRESHOLD,
    _HIGH_COMM_OVERHEAD_PCT,
    _STRAGGLER_SIGMA,
    _LOW_UTIL_PCT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_worker(
    worker_id: str = "w1",
    rank: int = 0,
    node_id: str = "node1",
    gpu_index: int = 0,
    iter_time: float = 100.0,
    compute_pct: float = 70.0,
    comm_pct: float = 20.0,
    idle_pct: float = 10.0,
    util_pct: float = 75.0,
    mem_mb: float = 4000.0,
    throughput: float = 1000.0,
    num_iter: int = 100,
) -> WorkerProfile:
    return WorkerProfile(
        worker_id=worker_id,
        worker_rank=rank,
        node_id=node_id,
        gpu_index=gpu_index,
        avg_iteration_time_ms=iter_time,
        compute_time_pct=compute_pct,
        communication_time_pct=comm_pct,
        idle_time_pct=idle_pct,
        gpu_utilization_pct=util_pct,
        gpu_memory_used_mb=mem_mb,
        throughput_samples_per_sec=throughput,
        num_iterations=num_iter,
    )


# ---------------------------------------------------------------------------
# WorkerProfile
# ---------------------------------------------------------------------------


class TestWorkerProfile:
    def test_creation(self):
        w = _make_worker()
        assert w.worker_id == "w1"
        assert w.avg_iteration_time_ms == 100.0

    def test_to_dict_from_dict_roundtrip(self):
        w = _make_worker(worker_id="w2", rank=3, gpu_index=2)
        d = w.to_dict()
        restored = WorkerProfile.from_dict(d)
        assert restored.worker_id == "w2"
        assert restored.worker_rank == 3
        assert restored.gpu_index == 2

    def test_from_dict_defaults(self):
        w = WorkerProfile.from_dict({})
        assert w.worker_id == ""
        assert w.worker_rank == 0


# ---------------------------------------------------------------------------
# CommunicationBottleneck
# ---------------------------------------------------------------------------


class TestCommunicationBottleneck:
    def test_creation(self):
        b = CommunicationBottleneck(
            bottleneck_type="straggler",
            severity="high",
            description="Worker w1 is slow.",
            affected_workers=["w1"],
            recommendation="Check thermal throttling.",
            metric_value=2.5,
        )
        assert b.bottleneck_type == "straggler"
        assert b.severity == "high"

    def test_to_dict_from_dict_roundtrip(self):
        b = CommunicationBottleneck(
            bottleneck_type="allreduce_slow",
            severity="medium",
            description="Slow allreduce.",
            affected_workers=["w1", "w2"],
            recommendation="Use gradient compression.",
            metric_value=35.0,
        )
        d = b.to_dict()
        restored = CommunicationBottleneck.from_dict(d)
        assert restored.bottleneck_type == "allreduce_slow"
        assert len(restored.affected_workers) == 2

    def test_from_dict_defaults(self):
        b = CommunicationBottleneck.from_dict({})
        assert b.bottleneck_type == ""
        assert b.affected_workers == []


# ---------------------------------------------------------------------------
# RayTrainProfileResult
# ---------------------------------------------------------------------------


class TestRayTrainProfileResult:
    def test_empty_result(self):
        r = RayTrainProfileResult(
            job_id="", job_name="", worker_profiles=[],
            communication_bottlenecks=[], num_workers=0,
            num_nodes=0, total_training_time_s=0.0,
            avg_throughput_samples_per_sec=0.0,
            communication_overhead_pct=0.0,
            load_balance_score=1.0, recommendations=[],
        )
        assert r.num_workers == 0

    def test_to_dict_from_dict_roundtrip(self):
        w = _make_worker()
        b = CommunicationBottleneck(
            bottleneck_type="straggler", severity="high",
            description="Slow worker.", affected_workers=["w1"],
            recommendation="Fix it.", metric_value=2.0,
        )
        r = RayTrainProfileResult(
            job_id="job1", job_name="Train",
            worker_profiles=[w],
            communication_bottlenecks=[b],
            num_workers=1, num_nodes=1,
            total_training_time_s=60.0,
            avg_throughput_samples_per_sec=1000.0,
            communication_overhead_pct=20.0,
            load_balance_score=0.9,
            recommendations=["Increase batch size."],
        )
        d = r.to_dict()
        restored = RayTrainProfileResult.from_dict(d)
        assert restored.job_id == "job1"
        assert len(restored.worker_profiles) == 1
        assert len(restored.communication_bottlenecks) == 1
        assert restored.recommendations == ["Increase batch size."]


# ---------------------------------------------------------------------------
# RayTrainIntegration — straggler detection
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationStragglers:
    def test_detects_straggler(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=100.0),
            _make_worker("w3", 2, iter_time=100.0),
            _make_worker("w4", 3, iter_time=300.0),  # straggler
        ]
        integration = RayTrainIntegration()
        bottlenecks = integration._detect_stragglers(workers)
        assert len(bottlenecks) >= 1
        assert bottlenecks[0].bottleneck_type == "straggler"
        assert "w4" in bottlenecks[0].affected_workers

    def test_no_stragglers_uniform(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=101.0),
            _make_worker("w3", 2, iter_time=99.0),
        ]
        integration = RayTrainIntegration()
        bottlenecks = integration._detect_stragglers(workers)
        assert len(bottlenecks) == 0

    def test_no_stragglers_single_worker(self):
        workers = [_make_worker("w1", 0, iter_time=100.0)]
        integration = RayTrainIntegration()
        bottlenecks = integration._detect_stragglers(workers)
        assert len(bottlenecks) == 0

    def test_no_stragglers_zero_times(self):
        workers = [
            _make_worker("w1", 0, iter_time=0.0),
            _make_worker("w2", 1, iter_time=0.0),
        ]
        integration = RayTrainIntegration()
        bottlenecks = integration._detect_stragglers(workers)
        assert len(bottlenecks) == 0


# ---------------------------------------------------------------------------
# RayTrainIntegration — load imbalance
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationImbalance:
    def test_detects_imbalance(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=200.0),  # 2x slower
        ]
        result = RayTrainIntegration._detect_load_imbalance(workers)
        assert result is not None
        assert result.bottleneck_type == "imbalanced"

    def test_no_imbalance_when_balanced(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=110.0),
        ]
        result = RayTrainIntegration._detect_load_imbalance(workers)
        assert result is None

    def test_no_imbalance_single_worker(self):
        workers = [_make_worker("w1", 0, iter_time=100.0)]
        result = RayTrainIntegration._detect_load_imbalance(workers)
        assert result is None


# ---------------------------------------------------------------------------
# RayTrainIntegration — high communication
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationHighComm:
    def test_detects_high_communication(self):
        workers = [
            _make_worker("w1", 0, comm_pct=50.0),
        ]
        bottlenecks = RayTrainIntegration._detect_high_communication(workers)
        assert len(bottlenecks) >= 1
        assert bottlenecks[0].bottleneck_type == "allreduce_slow"

    def test_no_detection_when_low_comm(self):
        workers = [
            _make_worker("w1", 0, comm_pct=10.0),
        ]
        bottlenecks = RayTrainIntegration._detect_high_communication(workers)
        assert len(bottlenecks) == 0


# ---------------------------------------------------------------------------
# RayTrainIntegration — low utilization
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationLowUtil:
    def test_detects_low_utilization(self):
        workers = [
            _make_worker("w1", 0, util_pct=20.0),
            _make_worker("w2", 1, util_pct=25.0),
            _make_worker("w3", 2, util_pct=30.0),
        ]
        bottlenecks = RayTrainIntegration._detect_low_utilization(workers)
        assert len(bottlenecks) >= 1
        assert bottlenecks[0].bottleneck_type == "bandwidth"

    def test_no_detection_when_util_high(self):
        workers = [
            _make_worker("w1", 0, util_pct=80.0),
            _make_worker("w2", 1, util_pct=85.0),
        ]
        bottlenecks = RayTrainIntegration._detect_low_utilization(workers)
        assert len(bottlenecks) == 0


# ---------------------------------------------------------------------------
# RayTrainIntegration — metrics computation
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationMetrics:
    def test_communication_overhead(self):
        workers = [
            _make_worker("w1", 0, comm_pct=20.0),
            _make_worker("w2", 1, comm_pct=30.0),
        ]
        overhead = RayTrainIntegration._compute_communication_overhead(workers)
        assert overhead == 25.0

    def test_communication_overhead_empty(self):
        assert RayTrainIntegration._compute_communication_overhead([]) == 0.0

    def test_load_balance_score_perfect(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=100.0),
        ]
        score = RayTrainIntegration._compute_load_balance_score(workers)
        assert score == 1.0

    def test_load_balance_score_imbalanced(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=200.0),
        ]
        score = RayTrainIntegration._compute_load_balance_score(workers)
        assert 0.0 < score < 1.0

    def test_load_balance_score_single_worker(self):
        workers = [_make_worker("w1", 0, iter_time=100.0)]
        score = RayTrainIntegration._compute_load_balance_score(workers)
        assert score == 1.0

    def test_avg_throughput(self):
        workers = [
            _make_worker("w1", 0, throughput=500.0),
            _make_worker("w2", 1, throughput=600.0),
        ]
        throughput = RayTrainIntegration._compute_avg_throughput(workers)
        assert throughput == 1100.0


# ---------------------------------------------------------------------------
# RayTrainIntegration — profile_from_workers
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationProfileFromWorkers:
    def test_basic_profiling(self):
        workers = [
            _make_worker("w1", 0, node_id="n1", iter_time=100.0, comm_pct=15.0),
            _make_worker("w2", 1, node_id="n1", iter_time=105.0, comm_pct=15.0),
            _make_worker("w3", 2, node_id="n2", iter_time=100.0, comm_pct=15.0),
            _make_worker("w4", 3, node_id="n2", iter_time=300.0, comm_pct=15.0),
        ]
        integration = RayTrainIntegration()
        result = integration.profile_from_workers(
            workers, job_id="test-job", job_name="Test",
            training_time_s=60.0,
        )
        assert result.job_id == "test-job"
        assert result.num_workers == 4
        assert result.num_nodes == 2
        assert len(result.communication_bottlenecks) >= 1  # straggler
        assert len(result.recommendations) > 0

    def test_empty_workers(self):
        integration = RayTrainIntegration()
        result = integration.profile_from_workers([])
        assert result.num_workers == 0
        assert result.communication_bottlenecks == []

    def test_well_balanced_workers(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0, comm_pct=10.0, util_pct=80.0),
            _make_worker("w2", 1, iter_time=100.0, comm_pct=10.0, util_pct=80.0),
        ]
        integration = RayTrainIntegration()
        result = integration.profile_from_workers(workers)
        assert result.load_balance_score == 1.0
        assert len(result.communication_bottlenecks) == 0


# ---------------------------------------------------------------------------
# RayTrainIntegration — recommendations
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationRecommendations:
    def test_recommendations_for_imbalanced(self):
        workers = [
            _make_worker("w1", 0, iter_time=100.0),
            _make_worker("w2", 1, iter_time=200.0),
        ]
        recs = RayTrainIntegration._generate_recommendations(
            workers, [], 10.0, 0.5,
        )
        assert any("balance" in r.lower() for r in recs)

    def test_recommendations_for_high_comm(self):
        recs = RayTrainIntegration._generate_recommendations(
            [], [], 30.0, 0.9,
        )
        assert any("communication" in r.lower() for r in recs)

    def test_recommendations_for_low_util(self):
        workers = [
            _make_worker("w1", 0, util_pct=30.0),
        ]
        recs = RayTrainIntegration._generate_recommendations(
            workers, [], 10.0, 1.0,
        )
        assert any("utilization" in r.lower() for r in recs)

    def test_no_bottleneck_recommendation(self):
        recs = RayTrainIntegration._generate_recommendations(
            [], [], 5.0, 1.0,
        )
        assert any("no major" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# RayTrainIntegration — detect without Ray
# ---------------------------------------------------------------------------


class TestRayTrainIntegrationDetect:
    def test_detect_without_ray(self):
        with patch("src.chamber.ray_integration._ray_available", False):
            integration = RayTrainIntegration()
            assert integration.detect_ray_train_job() is False

    def test_profile_without_ray(self):
        with patch("src.chamber.ray_integration._ray_available", False):
            integration = RayTrainIntegration()
            result = integration.profile()
            assert result.num_workers == 0

    def test_empty_result(self):
        integration = RayTrainIntegration()
        result = integration._empty_result()
        assert result.num_workers == 0
        assert result.load_balance_score == 1.0
