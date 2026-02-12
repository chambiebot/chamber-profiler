"""Tests for src.profiler.kernel_tracer."""

import pytest

from src.profiler.kernel_tracer import (
    KernelTracer,
    KernelRecord,
    KernelTraceResult,
    VALID_CATEGORIES,
)


# ---------------------------------------------------------------------------
# KernelRecord
# ---------------------------------------------------------------------------

class TestKernelRecord:
    def test_creation(self):
        record = KernelRecord(
            name="volta_sgemm_128x64_nn",
            duration_us=100.0,
            grid_size=(128, 1, 1),
            block_size=(256, 1, 1),
            shared_memory_bytes=4096,
            device_index=0,
            category="gemm",
        )
        assert record.name == "volta_sgemm_128x64_nn"
        assert record.category == "gemm"

    def test_invalid_category_defaults_to_other(self):
        record = KernelRecord(
            name="test",
            duration_us=10.0,
            grid_size=(1, 1, 1),
            block_size=(1, 1, 1),
            shared_memory_bytes=0,
            device_index=0,
            category="invalid_category",
        )
        assert record.category == "other"

    def test_to_dict_from_dict_roundtrip(self):
        record = KernelRecord(
            name="cutlass_gemm",
            duration_us=200.0,
            grid_size=(64, 32, 1),
            block_size=(128, 1, 1),
            shared_memory_bytes=8192,
            device_index=0,
            category="gemm",
            layer_name="encoder.layer.0",
            occupancy=0.75,
        )
        d = record.to_dict()
        restored = KernelRecord.from_dict(d)
        assert restored.name == record.name
        assert restored.grid_size == record.grid_size
        assert restored.block_size == record.block_size
        assert restored.occupancy == record.occupancy
        assert restored.layer_name == record.layer_name


# ---------------------------------------------------------------------------
# KernelTraceResult
# ---------------------------------------------------------------------------

class TestKernelTraceResult:
    def test_creation(self):
        result = KernelTraceResult(
            kernels=[],
            total_kernel_time_us=0.0,
            top_kernels=[],
            inefficient_kernels=[],
            category_breakdown={},
        )
        assert result.total_kernel_time_us == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        record = KernelRecord(
            name="test_kernel",
            duration_us=50.0,
            grid_size=(1, 1, 1),
            block_size=(32, 1, 1),
            shared_memory_bytes=0,
            device_index=0,
            category="other",
        )
        result = KernelTraceResult(
            kernels=[record],
            total_kernel_time_us=50.0,
            top_kernels=[record],
            inefficient_kernels=[],
            category_breakdown={"other": 50.0},
        )
        d = result.to_dict()
        restored = KernelTraceResult.from_dict(d)
        assert len(restored.kernels) == 1
        assert restored.total_kernel_time_us == 50.0
        assert restored.category_breakdown["other"] == 50.0


# ---------------------------------------------------------------------------
# KernelTracer
# ---------------------------------------------------------------------------

class TestKernelTracer:
    def test_start_stop_without_torch(self):
        """Tracer should work without torch - returns empty results."""
        tracer = KernelTracer(top_k=5)
        tracer.start()
        result = tracer.stop()
        assert isinstance(result, KernelTraceResult)
        assert result.total_kernel_time_us == 0.0
        assert result.kernels == []

    def test_stop_without_start(self):
        tracer = KernelTracer()
        result = tracer.stop()
        assert isinstance(result, KernelTraceResult)
        assert result.kernels == []

    def test_duplicate_start(self):
        tracer = KernelTracer()
        tracer.start()
        tracer.start()  # Should warn, not crash
        tracer.stop()

    def test_categorize_kernel_gemm(self):
        assert KernelTracer._categorize_kernel("volta_sgemm_128x64_nn") == "gemm"
        assert KernelTracer._categorize_kernel("cutlass_80_tensorop") == "gemm"
        assert KernelTracer._categorize_kernel("cublas_gemm_op") == "gemm"

    def test_categorize_kernel_conv(self):
        assert KernelTracer._categorize_kernel("cudnn_conv_fwd") == "conv"
        assert KernelTracer._categorize_kernel("winograd_kernel") == "conv"

    def test_categorize_kernel_attention(self):
        assert KernelTracer._categorize_kernel("flash_attn_fwd") == "attention"
        assert KernelTracer._categorize_kernel("fmha_kernel") == "attention"

    def test_categorize_kernel_communication(self):
        assert KernelTracer._categorize_kernel("ncclKernel_AllReduce") == "communication"

    def test_categorize_kernel_memory(self):
        assert KernelTracer._categorize_kernel("Memcpy_HtoD") == "memory"
        assert KernelTracer._categorize_kernel("fill_kernel") == "memory"

    def test_categorize_kernel_reduction(self):
        assert KernelTracer._categorize_kernel("layernorm_fwd") == "reduction"
        assert KernelTracer._categorize_kernel("softmax_kernel") == "reduction"

    def test_categorize_kernel_elementwise(self):
        assert KernelTracer._categorize_kernel("vectorized_elementwise") == "elementwise"
        assert KernelTracer._categorize_kernel("gelu_activation") == "elementwise"

    def test_categorize_kernel_other(self):
        assert KernelTracer._categorize_kernel("my_custom_kernel") == "other"

    def test_estimate_occupancy_zero_grid(self):
        record = KernelRecord(
            name="test", duration_us=1.0,
            grid_size=(0, 0, 0), block_size=(0, 0, 0),
            shared_memory_bytes=0, device_index=0, category="other",
        )
        assert KernelTracer._estimate_occupancy(record) is None

    def test_estimate_occupancy_large_grid(self):
        record = KernelRecord(
            name="test", duration_us=1.0,
            grid_size=(1000, 1, 1), block_size=(256, 1, 1),
            shared_memory_bytes=0, device_index=0, category="other",
        )
        occ = KernelTracer._estimate_occupancy(record)
        assert occ is not None
        assert 0.0 < occ <= 1.0

    def test_estimate_occupancy_clamped_to_one(self):
        record = KernelRecord(
            name="test", duration_us=1.0,
            grid_size=(10000, 100, 1), block_size=(1024, 1, 1),
            shared_memory_bytes=0, device_index=0, category="other",
        )
        occ = KernelTracer._estimate_occupancy(record)
        assert occ == 1.0

    def test_get_top_kernels(self):
        tracer = KernelTracer(top_k=2)
        tracer._kernels = [
            KernelRecord(name=f"k{i}", duration_us=float(i * 10),
                         grid_size=(1, 1, 1), block_size=(1, 1, 1),
                         shared_memory_bytes=0, device_index=0,
                         category="other")
            for i in range(5)
        ]
        top = tracer.get_top_kernels()
        assert len(top) == 2
        assert top[0].duration_us == 40.0
        assert top[1].duration_us == 30.0

    def test_get_inefficient_kernels(self):
        tracer = KernelTracer()
        mem_kernel = KernelRecord(
            name="memcpy", duration_us=10.0,
            grid_size=(1, 1, 1), block_size=(1, 1, 1),
            shared_memory_bytes=0, device_index=0, category="memory",
        )
        low_occ_kernel = KernelRecord(
            name="small", duration_us=10.0,
            grid_size=(1, 1, 1), block_size=(1, 1, 1),
            shared_memory_bytes=0, device_index=0, category="other",
            occupancy=0.1,
        )
        ok_kernel = KernelRecord(
            name="good", duration_us=10.0,
            grid_size=(1, 1, 1), block_size=(1, 1, 1),
            shared_memory_bytes=0, device_index=0, category="gemm",
            occupancy=0.8,
        )
        tracer._kernels = [mem_kernel, low_occ_kernel, ok_kernel]
        inefficient = tracer.get_inefficient_kernels()
        assert len(inefficient) == 2
        assert mem_kernel in inefficient
        assert low_occ_kernel in inefficient

    def test_extract_tuple_none(self):
        class Event:
            pass
        e = Event()
        assert KernelTracer._extract_tuple(e, "grid_size") == (0, 0, 0)

    def test_extract_tuple_list(self):
        class Event:
            grid_size = [128, 64, 1]
        assert KernelTracer._extract_tuple(Event(), "grid_size") == (128, 64, 1)

    def test_compute_category_breakdown(self):
        kernels = [
            KernelRecord(name="g1", duration_us=100.0,
                         grid_size=(1, 1, 1), block_size=(1, 1, 1),
                         shared_memory_bytes=0, device_index=0,
                         category="gemm"),
            KernelRecord(name="g2", duration_us=50.0,
                         grid_size=(1, 1, 1), block_size=(1, 1, 1),
                         shared_memory_bytes=0, device_index=0,
                         category="gemm"),
            KernelRecord(name="m1", duration_us=30.0,
                         grid_size=(1, 1, 1), block_size=(1, 1, 1),
                         shared_memory_bytes=0, device_index=0,
                         category="memory"),
        ]
        breakdown = KernelTracer._compute_category_breakdown(kernels)
        assert breakdown["gemm"] == 150.0
        assert breakdown["memory"] == 30.0
