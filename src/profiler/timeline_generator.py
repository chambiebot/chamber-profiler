"""Chrome trace format timeline generator.

Generates JSON timelines in the Chrome Trace Event Format that can be
loaded in ``chrome://tracing`` or Perfetto UI to visualize GPU compute,
data loading, and communication phases over time.

Reference: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult, GPUMetricSnapshot
from src.profiler.kernel_tracer import KernelTraceResult, KernelRecord
from src.profiler.communication_profiler import CommProfileResult, CollectiveOperation
from src.profiler.data_loading_profiler import DataLoadingResult, DataLoadingMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Process IDs for organizing trace events in the viewer.
_PID_GPU_METRICS = 1
_PID_KERNELS = 2
_PID_COMMUNICATION = 3
_PID_DATA_LOADING = 4
_PID_MEMORY = 5

# Thread IDs (tids) within each process.
_TID_UTILIZATION = 1
_TID_MEMORY_BW = 2
_TID_SM_ACTIVITY = 3
_TID_KERNEL_EXEC = 1
_TID_COMM_OPS = 1
_TID_BATCH_LOAD = 1


# ============================================================================
# Timeline Generator
# ============================================================================


class TimelineGenerator:
    """Generate Chrome trace format JSON timelines from profiling data.

    Usage::

        generator = TimelineGenerator(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            comm_result=comm_result,
        )
        generator.generate("timeline.json")

    The resulting JSON can be loaded in:
    - ``chrome://tracing`` (Chrome browser)
    - https://ui.perfetto.dev (Perfetto UI)
    """

    def __init__(
        self,
        gpu_result: Optional[GPUProfileResult] = None,
        kernel_result: Optional[KernelTraceResult] = None,
        comm_result: Optional[CommProfileResult] = None,
        data_result: Optional[DataLoadingResult] = None,
    ) -> None:
        self._gpu_result = gpu_result
        self._kernel_result = kernel_result
        self._comm_result = comm_result
        self._data_result = data_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> str:
        """Generate the timeline JSON and write to disk.

        Parameters
        ----------
        output_path:
            File path for the output JSON timeline.

        Returns
        -------
        str
            The output file path.
        """
        events = self.build_trace_events()
        trace = {"traceEvents": events}

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

        logger.info("Timeline written to %s (%d events)", output_path, len(events))
        return output_path

    def build_trace_events(self) -> List[Dict[str, Any]]:
        """Build the list of Chrome trace events from profiling data.

        Returns
        -------
        list[dict]
            A list of trace event dicts in Chrome Trace Event Format.
        """
        events: List[Dict[str, Any]] = []

        # Add process/thread metadata events for nice labels in the viewer.
        events.extend(self._build_metadata_events())

        # GPU metric snapshots -> counter events.
        if self._gpu_result is not None:
            events.extend(self._build_gpu_events())

        # Kernel records -> duration events.
        if self._kernel_result is not None:
            events.extend(self._build_kernel_events())

        # Communication ops -> duration events.
        if self._comm_result is not None:
            events.extend(self._build_comm_events())

        # Data loading -> duration events.
        if self._data_result is not None:
            events.extend(self._build_data_loading_events())

        return events

    # ------------------------------------------------------------------
    # Metadata events
    # ------------------------------------------------------------------

    def _build_metadata_events(self) -> List[Dict[str, Any]]:
        """Build process and thread name metadata events."""
        events: List[Dict[str, Any]] = []

        process_names = [
            (_PID_GPU_METRICS, "GPU Metrics"),
            (_PID_KERNELS, "CUDA Kernels"),
            (_PID_COMMUNICATION, "Communication"),
            (_PID_DATA_LOADING, "Data Loading"),
        ]

        for pid, name in process_names:
            events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": name},
            })

        thread_names = [
            (_PID_GPU_METRICS, _TID_UTILIZATION, "Utilization"),
            (_PID_GPU_METRICS, _TID_SM_ACTIVITY, "SM Activity"),
            (_PID_GPU_METRICS, _TID_MEMORY_BW, "Memory Bandwidth"),
            (_PID_KERNELS, _TID_KERNEL_EXEC, "Kernel Execution"),
            (_PID_COMMUNICATION, _TID_COMM_OPS, "Collective Ops"),
            (_PID_DATA_LOADING, _TID_BATCH_LOAD, "Batch Loading"),
        ]

        for pid, tid, name in thread_names:
            events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"name": name},
            })

        return events

    # ------------------------------------------------------------------
    # GPU metric events (counter type)
    # ------------------------------------------------------------------

    def _build_gpu_events(self) -> List[Dict[str, Any]]:
        """Build counter events from GPU metric snapshots."""
        if self._gpu_result is None or not self._gpu_result.snapshots:
            return []

        events: List[Dict[str, Any]] = []
        snapshots = self._gpu_result.snapshots

        base_ts = snapshots[0].timestamp if snapshots else 0.0

        for snap in snapshots:
            ts_us = (snap.timestamp - base_ts) * 1e6

            # GPU Utilization counter.
            events.append({
                "name": "GPU Utilization",
                "ph": "C",
                "ts": ts_us,
                "pid": _PID_GPU_METRICS,
                "tid": _TID_UTILIZATION,
                "args": {"utilization_pct": snap.utilization_pct},
            })

            # SM Activity counter.
            events.append({
                "name": "SM Activity",
                "ph": "C",
                "ts": ts_us,
                "pid": _PID_GPU_METRICS,
                "tid": _TID_SM_ACTIVITY,
                "args": {"sm_activity_pct": snap.sm_activity_pct},
            })

            # Memory Bandwidth counter.
            events.append({
                "name": "Memory Bandwidth",
                "ph": "C",
                "ts": ts_us,
                "pid": _PID_GPU_METRICS,
                "tid": _TID_MEMORY_BW,
                "args": {"memory_bandwidth_pct": snap.memory_bandwidth_pct},
            })

        return events

    # ------------------------------------------------------------------
    # Kernel events (duration type)
    # ------------------------------------------------------------------

    def _build_kernel_events(self) -> List[Dict[str, Any]]:
        """Build duration events from kernel trace records."""
        if self._kernel_result is None or not self._kernel_result.kernels:
            return []

        events: List[Dict[str, Any]] = []
        kernels = self._kernel_result.kernels

        # Kernels don't have individual timestamps -- lay them out
        # sequentially for visualization.
        running_ts_us = 0.0

        for kernel in kernels:
            events.append({
                "name": kernel.name,
                "cat": kernel.category,
                "ph": "X",  # Complete event
                "ts": running_ts_us,
                "dur": kernel.duration_us,
                "pid": _PID_KERNELS,
                "tid": _TID_KERNEL_EXEC,
                "args": {
                    "category": kernel.category,
                    "grid_size": list(kernel.grid_size),
                    "block_size": list(kernel.block_size),
                    "shared_memory_bytes": kernel.shared_memory_bytes,
                    "occupancy": kernel.occupancy,
                    "layer_name": kernel.layer_name,
                },
            })
            running_ts_us += kernel.duration_us

        return events

    # ------------------------------------------------------------------
    # Communication events (duration type)
    # ------------------------------------------------------------------

    def _build_comm_events(self) -> List[Dict[str, Any]]:
        """Build duration events from communication operations."""
        if self._comm_result is None or not self._comm_result.operations:
            return []

        events: List[Dict[str, Any]] = []
        ops = self._comm_result.operations

        base_ts = ops[0].timestamp if ops else 0.0

        for op in ops:
            ts_us = (op.timestamp - base_ts) * 1e6

            events.append({
                "name": op.name,
                "cat": "communication",
                "ph": "X",
                "ts": ts_us,
                "dur": op.duration_us,
                "pid": _PID_COMMUNICATION,
                "tid": _TID_COMM_OPS,
                "args": {
                    "data_size_bytes": op.data_size_bytes,
                    "bandwidth_gbps": op.bandwidth_gbps,
                    "src_rank": op.src_rank,
                    "dst_rank": op.dst_rank,
                },
            })

        return events

    # ------------------------------------------------------------------
    # Data loading events (duration type)
    # ------------------------------------------------------------------

    def _build_data_loading_events(self) -> List[Dict[str, Any]]:
        """Build duration events from data loading metrics."""
        if self._data_result is None or not self._data_result.metrics:
            return []

        events: List[Dict[str, Any]] = []
        metrics = self._data_result.metrics

        base_ts = metrics[0].timestamp if metrics else 0.0

        for m in metrics:
            ts_us = (m.timestamp - base_ts) * 1e6

            # Batch loading phase.
            events.append({
                "name": f"batch_{m.batch_index}_load",
                "cat": "data_loading",
                "ph": "X",
                "ts": ts_us,
                "dur": m.batch_load_time_ms * 1000.0,  # ms -> us
                "pid": _PID_DATA_LOADING,
                "tid": _TID_BATCH_LOAD,
                "args": {
                    "batch_index": m.batch_index,
                    "io_throughput_mbps": m.io_throughput_mbps,
                    "num_workers": m.num_workers_active,
                },
            })

            # GPU idle time (if significant).
            if m.gpu_idle_time_ms > 0.1:
                events.append({
                    "name": f"batch_{m.batch_index}_gpu_idle",
                    "cat": "idle",
                    "ph": "X",
                    "ts": ts_us + m.batch_load_time_ms * 1000.0,
                    "dur": m.gpu_idle_time_ms * 1000.0,
                    "pid": _PID_DATA_LOADING,
                    "tid": _TID_BATCH_LOAD,
                    "args": {
                        "batch_index": m.batch_index,
                        "gpu_idle_time_ms": m.gpu_idle_time_ms,
                    },
                })

        return events
