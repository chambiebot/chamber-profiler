"""Tests for src.profiler.timeline_generator."""

import json
import os
import tempfile

import pytest

from src.profiler.timeline_generator import (
    TimelineGenerator,
    _PID_GPU_METRICS,
    _PID_KERNELS,
    _PID_COMMUNICATION,
    _PID_DATA_LOADING,
    _TID_UTILIZATION,
    _TID_SM_ACTIVITY,
    _TID_MEMORY_BW,
    _TID_KERNEL_EXEC,
    _TID_COMM_OPS,
    _TID_BATCH_LOAD,
)
from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult, GPUMetricSnapshot
from src.profiler.kernel_tracer import KernelTraceResult, KernelRecord
from src.profiler.communication_profiler import CommProfileResult, CollectiveOperation
from src.profiler.data_loading_profiler import DataLoadingResult, DataLoadingMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gpu_result() -> GPUProfileResult:
    snaps = [
        GPUMetricSnapshot(
            timestamp=1000.0 + i * 0.1,
            gpu_index=0,
            utilization_pct=float(50 + i * 10),
            memory_used_mb=float(2000 + i * 500),
            memory_total_mb=8000.0,
            power_watts=200.0,
            temperature_c=70.0,
            sm_activity_pct=float(40 + i * 5),
            memory_bandwidth_pct=float(30 + i * 5),
            clock_speed_mhz=1500.0,
        )
        for i in range(3)
    ]
    return GPUProfileResult(duration_s=1.0, snapshots=snaps, gpu_count=1)


def _make_kernel_result() -> KernelTraceResult:
    kernels = [
        KernelRecord(
            name=f"kernel_{i}",
            duration_us=float(100 + i * 50),
            grid_size=(128, 1, 1),
            block_size=(256, 1, 1),
            shared_memory_bytes=4096,
            device_index=0,
            category="gemm" if i % 2 == 0 else "elementwise",
            layer_name=f"layer_{i}",
            occupancy=0.75,
        )
        for i in range(3)
    ]
    return KernelTraceResult(
        kernels=kernels,
        total_kernel_time_us=sum(k.duration_us for k in kernels),
        top_kernels=kernels[:2],
        inefficient_kernels=[],
        category_breakdown={"gemm": 200.0, "elementwise": 150.0},
    )


def _make_comm_result() -> CommProfileResult:
    ops = [
        CollectiveOperation(
            name=f"allreduce_{i}",
            duration_us=float(500 + i * 100),
            data_size_bytes=1024 * (i + 1),
            src_rank=0,
            dst_rank=1,
            timestamp=2000.0 + i * 0.5,
        )
        for i in range(2)
    ]
    return CommProfileResult(
        operations=ops,
        total_comm_time_us=sum(o.duration_us for o in ops),
        total_data_transferred_bytes=sum(o.data_size_bytes for o in ops),
        avg_bandwidth_gbps=10.0,
        comm_compute_overlap_pct=0.0,
        bottleneck_ops=[],
        straggler_ranks=[],
    )


def _make_data_loading_result() -> DataLoadingResult:
    metrics = [
        DataLoadingMetrics(
            batch_load_time_ms=float(5 + i),
            batch_process_time_ms=2.0,
            gpu_idle_time_ms=float(0.5 if i > 0 else 0.0),
            timestamp=3000.0 + i * 1.0,
            batch_index=i,
            num_workers_active=4,
            prefetch_buffer_size=2,
            io_throughput_mbps=500.0,
        )
        for i in range(3)
    ]
    return DataLoadingResult(
        metrics=metrics,
        avg_load_time_ms=6.0,
        avg_gpu_idle_time_ms=0.33,
        is_bottleneck=False,
        bottleneck_severity="none",
        recommendations=[],
        io_throughput_mbps=500.0,
    )


# ---------------------------------------------------------------------------
# Metadata events
# ---------------------------------------------------------------------------

class TestTimelineGeneratorMetadata:
    def test_metadata_events_always_present(self):
        gen = TimelineGenerator()
        events = gen.build_trace_events()
        # Should have process and thread metadata even with no data
        process_events = [e for e in events if e["ph"] == "M" and e["name"] == "process_name"]
        thread_events = [e for e in events if e["ph"] == "M" and e["name"] == "thread_name"]
        assert len(process_events) == 4  # GPU, Kernels, Communication, Data Loading
        assert len(thread_events) == 6

    def test_process_names(self):
        gen = TimelineGenerator()
        events = gen.build_trace_events()
        process_events = [e for e in events if e["ph"] == "M" and e["name"] == "process_name"]
        names = {e["args"]["name"] for e in process_events}
        assert "GPU Metrics" in names
        assert "CUDA Kernels" in names
        assert "Communication" in names
        assert "Data Loading" in names


# ---------------------------------------------------------------------------
# GPU events
# ---------------------------------------------------------------------------

class TestTimelineGeneratorGPU:
    def test_gpu_counter_events(self):
        gpu_result = _make_gpu_result()
        gen = TimelineGenerator(gpu_result=gpu_result)
        events = gen.build_trace_events()
        counter_events = [e for e in events if e["ph"] == "C"]
        # 3 snapshots * 3 counters (utilization, SM activity, memory BW)
        assert len(counter_events) == 9

    def test_gpu_utilization_counter(self):
        gpu_result = _make_gpu_result()
        gen = TimelineGenerator(gpu_result=gpu_result)
        events = gen.build_trace_events()
        util_events = [
            e for e in events
            if e["ph"] == "C" and e["name"] == "GPU Utilization"
        ]
        assert len(util_events) == 3
        assert all(e["pid"] == _PID_GPU_METRICS for e in util_events)
        assert all(e["tid"] == _TID_UTILIZATION for e in util_events)

    def test_gpu_events_timestamps_start_at_zero(self):
        gpu_result = _make_gpu_result()
        gen = TimelineGenerator(gpu_result=gpu_result)
        events = gen.build_trace_events()
        counter_events = [e for e in events if e["ph"] == "C"]
        timestamps = sorted(set(e["ts"] for e in counter_events))
        assert timestamps[0] == pytest.approx(0.0)

    def test_no_gpu_events_without_data(self):
        gen = TimelineGenerator(gpu_result=None)
        events = gen.build_trace_events()
        counter_events = [e for e in events if e["ph"] == "C"]
        assert len(counter_events) == 0


# ---------------------------------------------------------------------------
# Kernel events
# ---------------------------------------------------------------------------

class TestTimelineGeneratorKernels:
    def test_kernel_duration_events(self):
        kernel_result = _make_kernel_result()
        gen = TimelineGenerator(kernel_result=kernel_result)
        events = gen.build_trace_events()
        duration_events = [
            e for e in events
            if e["ph"] == "X" and e["pid"] == _PID_KERNELS
        ]
        assert len(duration_events) == 3

    def test_kernel_events_sequential(self):
        kernel_result = _make_kernel_result()
        gen = TimelineGenerator(kernel_result=kernel_result)
        events = gen.build_trace_events()
        kernel_events = sorted(
            [e for e in events if e["ph"] == "X" and e["pid"] == _PID_KERNELS],
            key=lambda e: e["ts"],
        )
        # Each kernel should start after the previous one ends
        for i in range(1, len(kernel_events)):
            prev_end = kernel_events[i - 1]["ts"] + kernel_events[i - 1]["dur"]
            assert kernel_events[i]["ts"] == pytest.approx(prev_end)

    def test_kernel_event_args(self):
        kernel_result = _make_kernel_result()
        gen = TimelineGenerator(kernel_result=kernel_result)
        events = gen.build_trace_events()
        kernel_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_KERNELS]
        for ev in kernel_events:
            assert "category" in ev["args"]
            assert "grid_size" in ev["args"]
            assert "block_size" in ev["args"]
            assert "occupancy" in ev["args"]

    def test_no_kernel_events_without_data(self):
        gen = TimelineGenerator(kernel_result=None)
        events = gen.build_trace_events()
        kernel_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_KERNELS]
        assert len(kernel_events) == 0


# ---------------------------------------------------------------------------
# Communication events
# ---------------------------------------------------------------------------

class TestTimelineGeneratorComm:
    def test_comm_duration_events(self):
        comm_result = _make_comm_result()
        gen = TimelineGenerator(comm_result=comm_result)
        events = gen.build_trace_events()
        comm_events = [
            e for e in events
            if e["ph"] == "X" and e["pid"] == _PID_COMMUNICATION
        ]
        assert len(comm_events) == 2

    def test_comm_event_category(self):
        comm_result = _make_comm_result()
        gen = TimelineGenerator(comm_result=comm_result)
        events = gen.build_trace_events()
        comm_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_COMMUNICATION]
        assert all(e["cat"] == "communication" for e in comm_events)

    def test_comm_event_args(self):
        comm_result = _make_comm_result()
        gen = TimelineGenerator(comm_result=comm_result)
        events = gen.build_trace_events()
        comm_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_COMMUNICATION]
        for ev in comm_events:
            assert "data_size_bytes" in ev["args"]
            assert "bandwidth_gbps" in ev["args"]

    def test_no_comm_events_without_data(self):
        gen = TimelineGenerator(comm_result=None)
        events = gen.build_trace_events()
        comm_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_COMMUNICATION]
        assert len(comm_events) == 0


# ---------------------------------------------------------------------------
# Data loading events
# ---------------------------------------------------------------------------

class TestTimelineGeneratorDataLoading:
    def test_data_loading_events(self):
        data_result = _make_data_loading_result()
        gen = TimelineGenerator(data_result=data_result)
        events = gen.build_trace_events()
        dl_events = [
            e for e in events
            if e["ph"] == "X" and e["pid"] == _PID_DATA_LOADING
        ]
        # 3 batch load events + 2 idle events (batch 1 and 2 have idle > 0.1)
        assert len(dl_events) >= 3

    def test_gpu_idle_events_only_when_significant(self):
        data_result = _make_data_loading_result()
        gen = TimelineGenerator(data_result=data_result)
        events = gen.build_trace_events()
        idle_events = [
            e for e in events
            if e["ph"] == "X" and e["pid"] == _PID_DATA_LOADING and "idle" in e["name"]
        ]
        # Batch 0 has 0.0ms idle (skipped), batches 1 and 2 have 0.5ms idle (> 0.1)
        assert len(idle_events) == 2

    def test_batch_load_event_args(self):
        data_result = _make_data_loading_result()
        gen = TimelineGenerator(data_result=data_result)
        events = gen.build_trace_events()
        load_events = [
            e for e in events
            if e["ph"] == "X" and e["pid"] == _PID_DATA_LOADING and "load" in e["name"]
        ]
        for ev in load_events:
            assert "batch_index" in ev["args"]
            assert "io_throughput_mbps" in ev["args"]

    def test_no_data_loading_events_without_data(self):
        gen = TimelineGenerator(data_result=None)
        events = gen.build_trace_events()
        dl_events = [e for e in events if e["ph"] == "X" and e["pid"] == _PID_DATA_LOADING]
        assert len(dl_events) == 0


# ---------------------------------------------------------------------------
# Full timeline generation
# ---------------------------------------------------------------------------

class TestTimelineGeneratorFull:
    def test_build_all_events(self):
        gen = TimelineGenerator(
            gpu_result=_make_gpu_result(),
            kernel_result=_make_kernel_result(),
            comm_result=_make_comm_result(),
            data_result=_make_data_loading_result(),
        )
        events = gen.build_trace_events()
        # Should have metadata + GPU + kernel + comm + data loading events
        assert len(events) > 20

    def test_generate_writes_file(self):
        gen = TimelineGenerator(
            gpu_result=_make_gpu_result(),
            kernel_result=_make_kernel_result(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "timeline.json")
            result_path = gen.generate(outpath)
            assert result_path == outpath
            assert os.path.exists(outpath)

            with open(outpath, "r") as f:
                trace = json.load(f)
            assert "traceEvents" in trace
            assert len(trace["traceEvents"]) > 0

    def test_generate_creates_parent_dirs(self):
        gen = TimelineGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "sub", "dir", "timeline.json")
            gen.generate(outpath)
            assert os.path.exists(outpath)

    def test_generate_valid_json(self):
        gen = TimelineGenerator(
            gpu_result=_make_gpu_result(),
            kernel_result=_make_kernel_result(),
            comm_result=_make_comm_result(),
            data_result=_make_data_loading_result(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "timeline.json")
            gen.generate(outpath)

            with open(outpath, "r") as f:
                trace = json.load(f)

            for event in trace["traceEvents"]:
                assert "ph" in event
                assert "pid" in event or event["ph"] == "M"

    def test_empty_timeline_still_has_metadata(self):
        gen = TimelineGenerator()
        events = gen.build_trace_events()
        assert len(events) == 10  # 4 process names + 6 thread names

    def test_chrome_trace_event_format_compliance(self):
        """Verify events follow Chrome Trace Event Format."""
        gen = TimelineGenerator(
            gpu_result=_make_gpu_result(),
            kernel_result=_make_kernel_result(),
        )
        events = gen.build_trace_events()
        for event in events:
            ph = event["ph"]
            if ph == "M":
                assert "name" in event
                assert "args" in event
            elif ph == "C":
                assert "ts" in event
                assert "args" in event
            elif ph == "X":
                assert "ts" in event
                assert "dur" in event
                assert event["dur"] >= 0
