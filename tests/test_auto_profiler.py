"""Tests for src.profiler.auto_profiler."""

import pytest

from src.profiler.auto_profiler import (
    AutoProfiler,
    AutoProfileResult,
    StepMetrics,
    Recommendation,
    DEFAULT_PROFILE_STEPS,
    DEFAULT_WARMUP_STEPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_profiler(
    profile_steps: int = 10,
    warmup_steps: int = 2,
    step_time: float = 0.001,
    gpu_util: float = 75.0,
    mem_used: float = 30000.0,
    mem_total: float = 80000.0,
) -> AutoProfileResult:
    """Run the auto profiler with manual step recording."""
    ap = AutoProfiler(profile_steps=profile_steps, warmup_steps=warmup_steps)
    total = warmup_steps + profile_steps
    for i in range(total):
        ap.record_step(
            step_index=i,
            wall_time_s=step_time,
            gpu_utilization_pct=gpu_util,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
        )
    return ap.report()


# ---------------------------------------------------------------------------
# Tests: StepMetrics
# ---------------------------------------------------------------------------


class TestStepMetrics:
    def test_round_trip(self):
        sm = StepMetrics(step=5, wall_time_s=0.1, gpu_utilization_pct=80.0,
                         memory_used_mb=4000.0, memory_total_mb=80000.0)
        d = sm.to_dict()
        sm2 = StepMetrics.from_dict(d)
        assert sm2.step == 5
        assert sm2.wall_time_s == pytest.approx(0.1)
        assert sm2.gpu_utilization_pct == pytest.approx(80.0)


class TestRecommendation:
    def test_round_trip(self):
        r = Recommendation(
            category="gpu_utilization", priority="high",
            title="Low util", description="GPU underused",
            estimated_impact="20% improvement",
        )
        d = r.to_dict()
        r2 = Recommendation.from_dict(d)
        assert r2.category == "gpu_utilization"
        assert r2.estimated_impact == "20% improvement"


# ---------------------------------------------------------------------------
# Tests: AutoProfiler
# ---------------------------------------------------------------------------


class TestAutoProfiler:
    def test_basic_profiling(self):
        result = _run_profiler(profile_steps=10, warmup_steps=2)
        assert result.total_steps_profiled == 10
        assert result.warmup_steps == 2
        assert result.avg_step_time_s > 0
        assert result.throughput_steps_per_sec > 0

    def test_warmup_excluded_from_stats(self):
        ap = AutoProfiler(profile_steps=5, warmup_steps=3)
        for i in range(8):
            # Warmup steps get a different time
            t = 1.0 if i < 3 else 0.01
            ap.record_step(i, wall_time_s=t, gpu_utilization_pct=70.0)
        report = ap.report()
        # Measurement steps are 3..7 (5 steps), avg should be ~0.01 not ~1.0
        assert report.total_steps_profiled == 5
        assert report.avg_step_time_s < 0.1

    def test_deactivates_after_all_steps(self):
        ap = AutoProfiler(profile_steps=5, warmup_steps=2)
        assert ap.is_active
        for i in range(7):
            ap.record_step(i, wall_time_s=0.01)
        assert not ap.is_active

    def test_extra_steps_ignored(self):
        ap = AutoProfiler(profile_steps=3, warmup_steps=1)
        for i in range(10):
            ap.record_step(i, wall_time_s=0.01)
        assert ap.steps_collected == 4  # 1 warmup + 3 measure

    def test_context_manager_api(self):
        ap = AutoProfiler(profile_steps=3, warmup_steps=1)
        for i in range(4):
            with ap.step(i):
                pass  # simulate work
        report = ap.report()
        assert report.total_steps_profiled == 3

    def test_callback_invoked(self):
        captured = []
        ap = AutoProfiler(
            profile_steps=3, warmup_steps=0,
            step_callback=lambda sm: captured.append(sm),
        )
        for i in range(3):
            ap.record_step(i, wall_time_s=0.01)
        assert len(captured) == 3
        assert all(isinstance(s, StepMetrics) for s in captured)

    def test_empty_report(self):
        ap = AutoProfiler(profile_steps=10, warmup_steps=5)
        report = ap.report()
        assert report.total_steps_profiled == 0
        assert report.recommendations == []

    def test_low_utilization_recommendation(self):
        result = _run_profiler(gpu_util=30.0)
        cats = [r.category for r in result.recommendations]
        assert "gpu_utilization" in cats

    def test_high_memory_recommendation(self):
        result = _run_profiler(mem_used=75000.0, mem_total=80000.0)
        cats = [r.category for r in result.recommendations]
        assert "memory" in cats
        # Should be critical priority
        mem_recs = [r for r in result.recommendations if r.category == "memory"]
        assert mem_recs[0].priority == "critical"

    def test_low_memory_recommendation(self):
        result = _run_profiler(mem_used=20000.0, mem_total=80000.0)
        cats = [r.category for r in result.recommendations]
        assert "memory" in cats
        mem_recs = [r for r in result.recommendations if r.category == "memory"]
        assert mem_recs[0].priority == "medium"

    def test_healthy_report(self):
        result = _run_profiler(gpu_util=80.0, mem_used=50000.0, mem_total=80000.0)
        assert len(result.recommendations) >= 1
        assert result.recommendations[0].category == "general"

    def test_slow_steps_detected(self):
        ap = AutoProfiler(profile_steps=10, warmup_steps=0)
        for i in range(10):
            t = 0.01 if i != 5 else 0.05  # step 5 is 5x slower
            ap.record_step(i, wall_time_s=t, gpu_utilization_pct=80.0,
                           memory_used_mb=50000, memory_total_mb=80000)
        report = ap.report()
        assert 5 in report.slow_steps

    def test_result_round_trip(self):
        result = _run_profiler()
        d = result.to_dict()
        r2 = AutoProfileResult.from_dict(d)
        assert r2.total_steps_profiled == result.total_steps_profiled
        assert len(r2.step_metrics) == len(result.step_metrics)

    def test_p95_step_time_variability(self):
        ap = AutoProfiler(profile_steps=20, warmup_steps=0)
        for i in range(20):
            # last 2 steps are slow
            t = 0.01 if i < 18 else 0.05
            ap.record_step(i, wall_time_s=t, gpu_utilization_pct=80.0,
                           memory_used_mb=50000, memory_total_mb=80000)
        report = ap.report()
        assert report.p95_step_time_s > report.median_step_time_s
