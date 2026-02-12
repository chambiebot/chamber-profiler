"""Tests for src.profiler.data_loading_profiler."""

import pytest

from src.profiler.data_loading_profiler import (
    DataLoadingProfiler,
    DataLoadingMetrics,
    DataLoadingResult,
    ProfiledDataLoader,
)


# ---------------------------------------------------------------------------
# DataLoadingMetrics
# ---------------------------------------------------------------------------

class TestDataLoadingMetrics:
    def test_creation(self):
        m = DataLoadingMetrics(
            batch_load_time_ms=10.0,
            batch_process_time_ms=50.0,
            gpu_idle_time_ms=5.0,
            timestamp=1000.0,
            batch_index=0,
            num_workers_active=4,
            prefetch_buffer_size=8,
            io_throughput_mbps=500.0,
        )
        assert m.batch_load_time_ms == 10.0
        assert m.batch_index == 0

    def test_to_dict_from_dict_roundtrip(self):
        m = DataLoadingMetrics(
            batch_load_time_ms=5.0, batch_process_time_ms=20.0,
            gpu_idle_time_ms=3.0, timestamp=1.0, batch_index=2,
            num_workers_active=8, prefetch_buffer_size=16,
            io_throughput_mbps=200.0,
        )
        d = m.to_dict()
        restored = DataLoadingMetrics.from_dict(d)
        assert restored == m


# ---------------------------------------------------------------------------
# DataLoadingResult
# ---------------------------------------------------------------------------

class TestDataLoadingResult:
    def test_creation(self):
        result = DataLoadingResult(
            metrics=[], avg_load_time_ms=0.0, avg_gpu_idle_time_ms=0.0,
            is_bottleneck=False, bottleneck_severity="none",
            recommendations=[], io_throughput_mbps=0.0,
        )
        assert result.is_bottleneck is False

    def test_to_dict_from_dict_roundtrip(self):
        m = DataLoadingMetrics(
            batch_load_time_ms=5.0, batch_process_time_ms=20.0,
            gpu_idle_time_ms=3.0, timestamp=1.0, batch_index=0,
            num_workers_active=4, prefetch_buffer_size=8,
            io_throughput_mbps=200.0,
        )
        result = DataLoadingResult(
            metrics=[m], avg_load_time_ms=5.0, avg_gpu_idle_time_ms=3.0,
            is_bottleneck=True, bottleneck_severity="moderate",
            recommendations=["Increase num_workers"],
            io_throughput_mbps=200.0,
        )
        d = result.to_dict()
        restored = DataLoadingResult.from_dict(d)
        assert len(restored.metrics) == 1
        assert restored.is_bottleneck is True
        assert restored.bottleneck_severity == "moderate"


# ---------------------------------------------------------------------------
# DataLoadingProfiler
# ---------------------------------------------------------------------------

class TestDataLoadingProfiler:
    def test_start_stop_empty(self):
        profiler = DataLoadingProfiler()
        profiler.start()
        result = profiler.stop()
        assert isinstance(result, DataLoadingResult)
        assert result.is_bottleneck is False
        assert result.bottleneck_severity == "none"

    def test_stop_without_start(self):
        profiler = DataLoadingProfiler()
        result = profiler.stop()
        assert isinstance(result, DataLoadingResult)
        assert result.metrics == []

    def test_duplicate_start(self):
        profiler = DataLoadingProfiler()
        profiler.start()
        profiler.start()  # Should warn, not crash
        profiler.stop()

    def test_record_batch(self):
        profiler = DataLoadingProfiler()
        profiler.start()
        profiler._record_batch(
            batch_index=0, load_time=10.0, process_time=50.0,
            gpu_idle_time=5.0, num_workers_active=4,
            prefetch_buffer_size=8, io_throughput_mbps=300.0,
        )
        result = profiler.stop()
        assert len(result.metrics) == 1
        assert result.avg_load_time_ms == 10.0

    def test_detect_bottleneck_no_data(self):
        profiler = DataLoadingProfiler()
        assert profiler.detect_bottleneck([]) is False

    def test_detect_bottleneck_is_bottleneck(self):
        profiler = DataLoadingProfiler()
        # High idle time relative to total -> bottleneck
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=100.0, batch_process_time_ms=100.0,
                gpu_idle_time_ms=80.0, timestamp=1.0, batch_index=0,
                num_workers_active=0, prefetch_buffer_size=0,
                io_throughput_mbps=10.0,
            ),
        ]
        assert profiler.detect_bottleneck(metrics) is True

    def test_detect_bottleneck_not_bottleneck(self):
        profiler = DataLoadingProfiler()
        # Low idle time relative to total -> not bottleneck
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=10.0, batch_process_time_ms=100.0,
                gpu_idle_time_ms=5.0, timestamp=1.0, batch_index=0,
                num_workers_active=8, prefetch_buffer_size=16,
                io_throughput_mbps=500.0,
            ),
        ]
        assert profiler.detect_bottleneck(metrics) is False

    def test_classify_severity_none(self):
        assert DataLoadingProfiler._classify_severity([]) == "none"

    def test_classify_severity_mild(self):
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=30.0, batch_process_time_ms=100.0,
                gpu_idle_time_ms=30.0, timestamp=1.0, batch_index=0,
                num_workers_active=4, prefetch_buffer_size=8,
                io_throughput_mbps=100.0,
            ),
        ]
        severity = DataLoadingProfiler._classify_severity(metrics)
        assert severity == "mild"

    def test_classify_severity_severe(self):
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=80.0, batch_process_time_ms=40.0,
                gpu_idle_time_ms=80.0, timestamp=1.0, batch_index=0,
                num_workers_active=0, prefetch_buffer_size=0,
                io_throughput_mbps=10.0,
            ),
        ]
        severity = DataLoadingProfiler._classify_severity(metrics)
        assert severity == "severe"

    def test_get_recommendations_empty(self):
        profiler = DataLoadingProfiler()
        assert profiler.get_recommendations([]) == []

    def test_get_recommendations_low_workers(self):
        profiler = DataLoadingProfiler()
        profiler._observed_num_workers = 0
        profiler._observed_pin_memory = False
        metrics = [
            DataLoadingMetrics(
                batch_load_time_ms=100.0, batch_process_time_ms=100.0,
                gpu_idle_time_ms=80.0, timestamp=1.0, batch_index=0,
                num_workers_active=0, prefetch_buffer_size=0,
                io_throughput_mbps=50.0,
            ),
        ]
        recs = profiler.get_recommendations(metrics)
        assert len(recs) > 0
        # Should recommend increasing num_workers
        assert any("num_workers" in r for r in recs)

    def test_wrap_dataloader(self):
        """Test wrapping a fake dataloader."""
        profiler = DataLoadingProfiler()
        profiler.start()

        class FakeLoader:
            num_workers = 2
            pin_memory = True
            prefetch_factor = 2

            def __iter__(self):
                return iter([1, 2, 3])

            def __len__(self):
                return 3

        loader = FakeLoader()
        wrapped = profiler.wrap_dataloader(loader)
        assert isinstance(wrapped, ProfiledDataLoader)
        assert profiler._observed_num_workers == 2
        assert profiler._observed_pin_memory is True

        # Iterate
        batches = list(wrapped)
        assert batches == [1, 2, 3]

        result = profiler.stop()
        assert len(result.metrics) == 3
