"""Tests for src.profiler.communication_profiler."""

import pytest

from src.profiler.communication_profiler import (
    CommunicationProfiler,
    CollectiveOperation,
    CommProfileResult,
    _estimate_data_size,
    _extract_ranks,
    _optional_int,
)


# ---------------------------------------------------------------------------
# CollectiveOperation
# ---------------------------------------------------------------------------

class TestCollectiveOperation:
    def test_bandwidth_calculation(self):
        op = CollectiveOperation(
            name="all_reduce",
            duration_us=1000.0,  # 1 ms
            data_size_bytes=1_000_000,  # 1 MB
            src_rank=0,
            dst_rank=None,
            timestamp=1.0,
        )
        # bandwidth = (1e6 / 1000) * (8/1000) = 8 Gbps
        assert op.bandwidth_gbps == pytest.approx(8.0, abs=0.01)

    def test_bandwidth_zero_duration(self):
        op = CollectiveOperation(
            name="barrier",
            duration_us=0.0,
            data_size_bytes=0,
            src_rank=None,
            dst_rank=None,
            timestamp=1.0,
        )
        assert op.bandwidth_gbps == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        op = CollectiveOperation(
            name="all_gather",
            duration_us=500.0,
            data_size_bytes=50000,
            src_rank=0,
            dst_rank=1,
            timestamp=1.0,
        )
        d = op.to_dict()
        restored = CollectiveOperation.from_dict(d)
        assert restored.name == op.name
        assert restored.duration_us == op.duration_us
        assert restored.src_rank == op.src_rank
        assert restored.dst_rank == op.dst_rank


# ---------------------------------------------------------------------------
# CommProfileResult
# ---------------------------------------------------------------------------

class TestCommProfileResult:
    def test_creation(self):
        result = CommProfileResult(
            operations=[],
            total_comm_time_us=0.0,
            total_data_transferred_bytes=0,
            avg_bandwidth_gbps=0.0,
            comm_compute_overlap_pct=0.0,
            bottleneck_ops=[],
            straggler_ranks=[],
        )
        assert result.total_comm_time_us == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        op = CollectiveOperation(
            name="all_reduce", duration_us=100.0,
            data_size_bytes=10000, src_rank=0, dst_rank=None,
            timestamp=1.0,
        )
        result = CommProfileResult(
            operations=[op],
            total_comm_time_us=100.0,
            total_data_transferred_bytes=10000,
            avg_bandwidth_gbps=0.8,
            comm_compute_overlap_pct=50.0,
            bottleneck_ops=[op],
            straggler_ranks=[2, 3],
        )
        d = result.to_dict()
        restored = CommProfileResult.from_dict(d)
        assert len(restored.operations) == 1
        assert restored.straggler_ranks == [2, 3]


# ---------------------------------------------------------------------------
# CommunicationProfiler
# ---------------------------------------------------------------------------

class TestCommunicationProfiler:
    def test_start_stop_without_dist(self):
        """Profiler should work without torch.distributed."""
        profiler = CommunicationProfiler()
        profiler.start()
        result = profiler.stop()
        assert isinstance(result, CommProfileResult)
        assert result.total_comm_time_us == 0.0

    def test_stop_without_start(self):
        profiler = CommunicationProfiler()
        result = profiler.stop()
        assert isinstance(result, CommProfileResult)
        assert result.operations == []

    def test_duplicate_start(self):
        profiler = CommunicationProfiler()
        profiler.start()
        profiler.start()  # Should warn, not crash
        profiler.stop()

    def test_get_comm_compute_overlap_no_ops(self):
        profiler = CommunicationProfiler()
        assert profiler.get_comm_compute_overlap() == 0.0

    def test_get_comm_compute_overlap_no_compute_events(self):
        profiler = CommunicationProfiler()
        profiler._operations = [
            CollectiveOperation(
                name="all_reduce", duration_us=100.0,
                data_size_bytes=1000, src_rank=0, dst_rank=None,
                timestamp=1.0,
            ),
        ]
        assert profiler.get_comm_compute_overlap() == 0.0

    def test_record_compute_event(self):
        profiler = CommunicationProfiler()
        profiler.record_compute_event(1.0, 2.0)
        assert len(profiler._compute_events) == 1
        # Invalid event (end before start) should be ignored
        profiler.record_compute_event(3.0, 2.0)
        assert len(profiler._compute_events) == 1

    def test_get_bandwidth_utilization_empty(self):
        profiler = CommunicationProfiler()
        result = profiler.get_bandwidth_utilization()
        assert result["avg_bandwidth_gbps"] == 0.0
        assert result["utilization_pct"] == 0.0

    def test_get_bandwidth_utilization_with_ops(self):
        profiler = CommunicationProfiler()
        profiler._operations = [
            CollectiveOperation(
                name="all_reduce", duration_us=1000.0,
                data_size_bytes=1_000_000, src_rank=0, dst_rank=None,
                timestamp=1.0,
            ),
        ]
        result = profiler.get_bandwidth_utilization(theoretical_bw_gbps=100.0)
        assert result["avg_bandwidth_gbps"] > 0
        assert result["utilization_pct"] > 0

    def test_detect_stragglers_empty(self):
        profiler = CommunicationProfiler()
        assert profiler.detect_stragglers() == []

    def test_detect_stragglers_uniform(self):
        profiler = CommunicationProfiler()
        # All ranks have same duration - no stragglers
        for rank in range(4):
            profiler._operations.append(
                CollectiveOperation(
                    name="all_reduce", duration_us=100.0,
                    data_size_bytes=1000, src_rank=rank, dst_rank=None,
                    timestamp=1.0,
                )
            )
        assert profiler.detect_stragglers() == []

    def test_find_bottleneck_ops_empty(self):
        profiler = CommunicationProfiler()
        assert profiler._find_bottleneck_ops([]) == []

    def test_find_bottleneck_ops(self):
        profiler = CommunicationProfiler()
        ops = [
            CollectiveOperation(
                name="all_reduce", duration_us=100.0,
                data_size_bytes=1000, src_rank=0, dst_rank=None,
                timestamp=1.0,
            ),
            CollectiveOperation(
                name="all_reduce", duration_us=100.0,
                data_size_bytes=1000, src_rank=0, dst_rank=None,
                timestamp=2.0,
            ),
            CollectiveOperation(
                name="all_reduce_slow", duration_us=1000.0,
                data_size_bytes=1000, src_rank=0, dst_rank=None,
                timestamp=3.0,
            ),
        ]
        bottlenecks = profiler._find_bottleneck_ops(ops)
        assert len(bottlenecks) == 1
        assert bottlenecks[0].name == "all_reduce_slow"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_estimate_data_size_no_tensors(self):
        assert _estimate_data_size(("hello", 42), {"key": "value"}) == 0

    def test_extract_ranks_broadcast(self):
        src, dst = _extract_ranks("broadcast", (None, 3), {})
        assert src == 3
        assert dst is None

    def test_extract_ranks_reduce(self):
        src, dst = _extract_ranks("reduce", (None, 5), {})
        assert src is None
        assert dst == 5

    def test_extract_ranks_kwargs(self):
        src, dst = _extract_ranks("all_reduce", (), {"src": 1, "dst": 2})
        assert src == 1
        assert dst == 2

    def test_optional_int(self):
        assert _optional_int(None) is None
        assert _optional_int(42) == 42
        assert _optional_int("3") == 3
        assert _optional_int("invalid") is None
