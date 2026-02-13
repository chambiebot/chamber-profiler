"""Tests for src.profiler.io_profiler."""

import os
import tempfile

import pytest

from src.profiler.io_profiler import (
    IOOperation,
    IOProfileResult,
    IOProfiler,
)


# ---------------------------------------------------------------------------
# IOOperation
# ---------------------------------------------------------------------------

class TestIOOperation:
    def test_creation(self):
        op = IOOperation(
            operation="read", path="/data/train.bin",
            size_mb=1024.0, duration_sec=2.0,
            throughput_mbps=512.0, timestamp=1000.0,
        )
        assert op.operation == "read"
        assert op.throughput_mbps == 512.0

    def test_to_dict_from_dict_roundtrip(self):
        op = IOOperation(
            operation="write", path="/tmp/test.bin",
            size_mb=50.0, duration_sec=0.5,
            throughput_mbps=100.0, timestamp=2000.0,
        )
        d = op.to_dict()
        restored = IOOperation.from_dict(d)
        assert restored.operation == "write"
        assert restored.size_mb == 50.0

    def test_frozen(self):
        op = IOOperation(
            operation="read", path="", size_mb=0.0,
            duration_sec=0.0, throughput_mbps=0.0, timestamp=0.0,
        )
        with pytest.raises(AttributeError):
            op.size_mb = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IOProfileResult
# ---------------------------------------------------------------------------

class TestIOProfileResult:
    def test_to_dict_from_dict_roundtrip(self):
        op = IOOperation(
            operation="read", path="/data/x.bin",
            size_mb=100.0, duration_sec=1.0,
            throughput_mbps=100.0, timestamp=1000.0,
        )
        result = IOProfileResult(
            operations=[op],
            avg_read_throughput_mbps=100.0,
            avg_write_throughput_mbps=0.0,
            avg_save_throughput_mbps=0.0,
            avg_load_throughput_mbps=0.0,
            total_io_time_sec=1.0,
            is_io_bottleneck=True,
            recommendations=["Use NVMe"],
        )
        d = result.to_dict()
        restored = IOProfileResult.from_dict(d)
        assert len(restored.operations) == 1
        assert restored.is_io_bottleneck is True
        assert len(restored.recommendations) == 1

    def test_empty_roundtrip(self):
        result = IOProfileResult(
            operations=[], avg_read_throughput_mbps=0.0,
            avg_write_throughput_mbps=0.0, avg_save_throughput_mbps=0.0,
            avg_load_throughput_mbps=0.0, total_io_time_sec=0.0,
            is_io_bottleneck=False, recommendations=[],
        )
        d = result.to_dict()
        restored = IOProfileResult.from_dict(d)
        assert restored.operations == []


# ---------------------------------------------------------------------------
# IOProfiler — file operations
# ---------------------------------------------------------------------------

class TestIOProfilerFileOps:
    def test_profile_file_read(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test_read.bin"
        test_file.write_bytes(b"\x00" * (1024 * 1024))  # 1 MB

        profiler = IOProfiler()
        op = profiler.profile_file_read(str(test_file))
        assert op.operation == "read"
        assert op.size_mb > 0.9  # ~1 MB
        assert op.throughput_mbps > 0
        assert op.duration_sec > 0

    def test_profile_file_read_missing_file(self):
        profiler = IOProfiler()
        op = profiler.profile_file_read("/nonexistent/file.bin")
        assert op.operation == "read"
        assert op.size_mb == 0.0
        assert op.throughput_mbps == 0.0

    def test_profile_file_write(self, tmp_path):
        test_file = tmp_path / "test_write.bin"

        profiler = IOProfiler()
        op = profiler.profile_file_write(str(test_file), size_mb=1.0)
        assert op.operation == "write"
        assert op.size_mb == 1.0
        assert op.throughput_mbps > 0
        assert test_file.exists()
        # Verify actual file size
        assert test_file.stat().st_size == 1024 * 1024

    def test_profile_file_write_invalid_path(self):
        profiler = IOProfiler()
        op = profiler.profile_file_write("/nonexistent/dir/file.bin", size_mb=1.0)
        assert op.operation == "write"
        assert op.throughput_mbps == 0.0

    def test_profile_custom_operation(self):
        profiler = IOProfiler()

        data = []
        def custom_io():
            data.extend(range(1000))

        op = profiler.profile_operation(
            name="custom_read", fn=custom_io,
            size_mb=10.0, path="/data/custom",
        )
        assert op.operation == "custom_read"
        assert op.size_mb == 10.0
        assert op.throughput_mbps > 0


# ---------------------------------------------------------------------------
# IOProfiler — results and analysis
# ---------------------------------------------------------------------------

class TestIOProfilerResults:
    def test_empty_results(self):
        profiler = IOProfiler()
        result = profiler.get_results()
        assert result.operations == []
        assert result.is_io_bottleneck is False

    def test_aggregated_results(self, tmp_path):
        profiler = IOProfiler()

        # Do a read and a write with enough data to measure
        f1 = tmp_path / "read.bin"
        f1.write_bytes(b"\x00" * (4 * 1024 * 1024))  # 4 MB
        profiler.profile_file_read(str(f1))

        f2 = tmp_path / "write.bin"
        profiler.profile_file_write(str(f2), size_mb=4.0)

        result = profiler.get_results()
        assert len(result.operations) == 2
        assert result.avg_read_throughput_mbps > 0
        assert result.avg_write_throughput_mbps > 0
        assert result.total_io_time_sec >= 0  # may be very fast on SSD

    def test_reset(self, tmp_path):
        profiler = IOProfiler()
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00" * 1024)
        profiler.profile_file_read(str(f))
        assert len(profiler.get_results().operations) == 1

        profiler.reset()
        assert len(profiler.get_results().operations) == 0

    def test_bottleneck_detection_slow_read(self):
        profiler = IOProfiler()
        # Manually add a slow read operation
        profiler._operations.append(IOOperation(
            operation="read", path="/data/slow.bin",
            size_mb=100.0, duration_sec=2.0,
            throughput_mbps=50.0,  # below 200 MB/s threshold
            timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert result.is_io_bottleneck is True

    def test_bottleneck_detection_fast_read(self):
        profiler = IOProfiler()
        profiler._operations.append(IOOperation(
            operation="read", path="/data/fast.bin",
            size_mb=100.0, duration_sec=0.25,
            throughput_mbps=400.0,  # above threshold
            timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert result.is_io_bottleneck is False

    def test_bottleneck_detection_slow_write(self):
        profiler = IOProfiler()
        profiler._operations.append(IOOperation(
            operation="write", path="/tmp/slow.bin",
            size_mb=100.0, duration_sec=5.0,
            throughput_mbps=20.0,  # below 100 MB/s threshold
            timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert result.is_io_bottleneck is True


# ---------------------------------------------------------------------------
# IOProfiler — recommendations
# ---------------------------------------------------------------------------

class TestIOProfilerRecommendations:
    def test_slow_read_recommendation(self):
        profiler = IOProfiler()
        profiler._operations.append(IOOperation(
            operation="read", path="/data/x.bin",
            size_mb=100.0, duration_sec=2.0,
            throughput_mbps=50.0, timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert any("read" in r.lower() or "Read" in r for r in result.recommendations)

    def test_slow_write_recommendation(self):
        profiler = IOProfiler()
        profiler._operations.append(IOOperation(
            operation="write", path="/tmp/x.bin",
            size_mb=100.0, duration_sec=5.0,
            throughput_mbps=20.0, timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert any("write" in r.lower() or "Write" in r for r in result.recommendations)

    def test_high_total_io_time_recommendation(self):
        profiler = IOProfiler()
        # Add operations with large total time
        for i in range(10):
            profiler._operations.append(IOOperation(
                operation="read", path=f"/data/{i}.bin",
                size_mb=1000.0, duration_sec=10.0,
                throughput_mbps=100.0, timestamp=float(i),
            ))
        result = profiler.get_results()
        any_total_time = any("total" in r.lower() or "I/O time" in r for r in result.recommendations)
        assert any_total_time

    def test_no_recommendations_fast_io(self):
        profiler = IOProfiler()
        profiler._operations.append(IOOperation(
            operation="read", path="/data/x.bin",
            size_mb=100.0, duration_sec=0.1,
            throughput_mbps=1000.0, timestamp=1000.0,
        ))
        profiler._operations.append(IOOperation(
            operation="write", path="/tmp/x.bin",
            size_mb=100.0, duration_sec=0.2,
            throughput_mbps=500.0, timestamp=1000.0,
        ))
        result = profiler.get_results()
        assert result.recommendations == []
