"""Tests for src.profiler.batch_size_optimizer."""

import pytest

from src.profiler.batch_size_optimizer import (
    BatchSizeTrial,
    BatchSizeResult,
    BatchSizeOptimizer,
)


# ---------------------------------------------------------------------------
# BatchSizeTrial
# ---------------------------------------------------------------------------

class TestBatchSizeTrial:
    def test_creation(self):
        trial = BatchSizeTrial(
            batch_size=32,
            throughput_samples_per_sec=1000.0,
            avg_step_time_ms=32.0,
            peak_memory_mb=4096.0,
            avg_gpu_utilization_pct=85.0,
            oom=False,
        )
        assert trial.batch_size == 32
        assert trial.oom is False

    def test_oom_trial(self):
        trial = BatchSizeTrial(
            batch_size=256,
            throughput_samples_per_sec=0.0,
            avg_step_time_ms=0.0,
            peak_memory_mb=0.0,
            avg_gpu_utilization_pct=0.0,
            oom=True,
            error="CUDA out of memory",
        )
        assert trial.oom is True
        assert trial.error is not None

    def test_to_dict_from_dict_roundtrip(self):
        trial = BatchSizeTrial(
            batch_size=64,
            throughput_samples_per_sec=2000.0,
            avg_step_time_ms=32.0,
            peak_memory_mb=8000.0,
            avg_gpu_utilization_pct=90.0,
            oom=False,
        )
        d = trial.to_dict()
        restored = BatchSizeTrial.from_dict(d)
        assert restored.batch_size == 64
        assert restored.throughput_samples_per_sec == 2000.0

    def test_frozen(self):
        trial = BatchSizeTrial(
            batch_size=8, throughput_samples_per_sec=100.0,
            avg_step_time_ms=80.0, peak_memory_mb=1000.0,
            avg_gpu_utilization_pct=50.0, oom=False,
        )
        with pytest.raises(AttributeError):
            trial.batch_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BatchSizeResult
# ---------------------------------------------------------------------------

class TestBatchSizeResult:
    def test_to_dict_from_dict_roundtrip(self):
        trial = BatchSizeTrial(
            batch_size=32, throughput_samples_per_sec=1500.0,
            avg_step_time_ms=21.3, peak_memory_mb=6000.0,
            avg_gpu_utilization_pct=80.0, oom=False,
        )
        result = BatchSizeResult(
            trials=[trial],
            optimal_batch_size=32,
            optimal_throughput=1500.0,
            max_safe_batch_size=32,
            memory_limit_mb=16000.0,
            recommendations=["Use batch_size=32"],
        )
        d = result.to_dict()
        restored = BatchSizeResult.from_dict(d)
        assert len(restored.trials) == 1
        assert restored.optimal_batch_size == 32
        assert restored.memory_limit_mb == 16000.0

    def test_empty_roundtrip(self):
        result = BatchSizeResult(
            trials=[], optimal_batch_size=0,
            optimal_throughput=0.0, max_safe_batch_size=0,
            memory_limit_mb=None, recommendations=[],
        )
        d = result.to_dict()
        restored = BatchSizeResult.from_dict(d)
        assert restored.trials == []
        assert restored.memory_limit_mb is None


# ---------------------------------------------------------------------------
# BatchSizeOptimizer
# ---------------------------------------------------------------------------

class TestBatchSizeOptimizer:
    def test_no_train_fn_returns_empty(self):
        optimizer = BatchSizeOptimizer()
        result = optimizer.run()
        assert result.optimal_batch_size == 0
        assert result.trials == []

    def test_find_optimal_from_trials(self):
        trials = [
            BatchSizeTrial(8, 500.0, 16.0, 2000.0, 60.0, False),
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 1200.0, 26.7, 6000.0, 80.0, False),
            BatchSizeTrial(64, 1100.0, 58.2, 10000.0, 75.0, False),
        ]
        optimizer = BatchSizeOptimizer()
        bs, tp = optimizer.find_optimal(trials)
        assert bs == 32
        assert tp == 1200.0

    def test_find_optimal_all_oom(self):
        trials = [
            BatchSizeTrial(8, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
            BatchSizeTrial(16, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
        ]
        optimizer = BatchSizeOptimizer()
        bs, tp = optimizer.find_optimal(trials)
        assert bs == 0
        assert tp == 0.0

    def test_find_max_safe_batch_size(self):
        trials = [
            BatchSizeTrial(8, 500.0, 16.0, 2000.0, 60.0, False),
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 1200.0, 26.7, 12000.0, 80.0, False),  # near limit
            BatchSizeTrial(64, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
        ]
        optimizer = BatchSizeOptimizer(memory_limit_mb=16000.0, memory_headroom_pct=10.0)
        max_safe = optimizer.find_max_safe_batch_size(trials)
        # 16000 * 0.9 = 14400 MB headroom limit
        # bs=32 uses 12000 MB < 14400, so it's safe
        assert max_safe == 32

    def test_find_max_safe_batch_size_no_memory_limit(self):
        trials = [
            BatchSizeTrial(8, 500.0, 16.0, 2000.0, 60.0, False),
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
        ]
        optimizer = BatchSizeOptimizer()
        max_safe = optimizer.find_max_safe_batch_size(trials)
        assert max_safe == 16  # largest successful

    def test_analyze_trials(self):
        trials = [
            BatchSizeTrial(8, 500.0, 16.0, 2000.0, 60.0, False),
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
        ]
        optimizer = BatchSizeOptimizer(memory_limit_mb=16000.0)
        result = optimizer.analyze_trials(trials)
        assert result.optimal_batch_size == 16
        assert result.optimal_throughput == 900.0
        assert len(result.recommendations) > 0

    def test_recommendations_oom(self):
        trials = [
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
        ]
        optimizer = BatchSizeOptimizer()
        result = optimizer.analyze_trials(trials)
        # Should mention OOM
        any_oom = any("OOM" in r or "oom" in r.lower() for r in result.recommendations)
        assert any_oom

    def test_recommendations_plateau(self):
        trials = [
            BatchSizeTrial(8, 500.0, 16.0, 2000.0, 60.0, False),
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 910.0, 35.2, 6000.0, 72.0, False),  # ~1% improvement
        ]
        optimizer = BatchSizeOptimizer()
        result = optimizer.analyze_trials(trials)
        any_plateau = any("plateau" in r.lower() for r in result.recommendations)
        assert any_plateau

    def test_recommendations_gradient_accumulation(self):
        trials = [
            BatchSizeTrial(16, 900.0, 17.8, 3500.0, 70.0, False),
            BatchSizeTrial(32, 1200.0, 26.7, 6000.0, 80.0, False),
            BatchSizeTrial(64, 0.0, 0.0, 0.0, 0.0, True, "OOM"),
        ]
        optimizer = BatchSizeOptimizer()
        result = optimizer.analyze_trials(trials)
        any_accum = any("accumulation" in r.lower() for r in result.recommendations)
        assert any_accum

    def test_run_with_simple_fn(self):
        """Test run() with a trivial train step function."""
        call_count = 0

        def simple_train_step(batch_size: int) -> int:
            nonlocal call_count
            call_count += 1
            return batch_size

        optimizer = BatchSizeOptimizer(
            train_step_fn=simple_train_step,
            batch_sizes=[4, 8],
            warmup_steps=1,
            measure_steps=2,
        )
        result = optimizer.run()
        assert len(result.trials) == 2
        assert result.optimal_batch_size > 0
        assert call_count > 0

    def test_run_with_oom_fn(self):
        """Test that OOM during training is handled gracefully."""

        def oom_train_step(batch_size: int) -> int:
            if batch_size >= 16:
                raise RuntimeError("CUDA out of memory")
            return batch_size

        optimizer = BatchSizeOptimizer(
            train_step_fn=oom_train_step,
            batch_sizes=[8, 16, 32],
            warmup_steps=1,
            measure_steps=2,
        )
        result = optimizer.run()
        successful = [t for t in result.trials if not t.oom]
        oom_trials = [t for t in result.trials if t.oom]
        assert len(successful) == 1  # only bs=8
        assert len(oom_trials) >= 1  # at least bs=16

    def test_run_oom_during_warmup(self):
        """OOM during warmup should be caught too."""

        def oom_warmup(batch_size: int) -> int:
            raise RuntimeError("CUDA out of memory")

        optimizer = BatchSizeOptimizer(
            train_step_fn=oom_warmup,
            batch_sizes=[8],
            warmup_steps=1,
            measure_steps=2,
        )
        result = optimizer.run()
        assert len(result.trials) == 1
        assert result.trials[0].oom is True

    def test_empty_trials_result(self):
        result = BatchSizeOptimizer._empty_result()
        assert result.optimal_batch_size == 0
        assert result.trials == []
