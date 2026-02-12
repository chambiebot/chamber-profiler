"""High-level Python SDK for chamber-profiler.

Provides a simple, user-facing API to profile ML training code with
minimal integration effort.  Users can profile entire training loops,
individual functions, or arbitrary code blocks via context managers.

Example::

    from src.sdk import ChamberProfiler

    profiler = ChamberProfiler()

    with profiler.profile():
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()

    profiler.report()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from src.profiler.gpu_profiler import GPUProfiler, ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTracer, KernelTraceResult
from src.profiler.memory_analyzer import MemoryAnalyzer, MemoryAnalysisResult
from src.profiler.communication_profiler import CommunicationProfiler, CommProfileResult
from src.profiler.data_loading_profiler import DataLoadingProfiler, DataLoadingResult
from src.analysis.bottleneck_detector import BottleneckDetector, PerformanceSummary
from src.analysis.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ChamberProfiler:
    """One-stop GPU profiling SDK for ML training.

    Orchestrates all profiler sub-modules (GPU metrics, kernel tracing,
    memory analysis, communication profiling, data loading profiling)
    and exposes simple ``start()``/``stop()``/``profile()`` helpers.

    Parameters
    ----------
    gpu : bool
        Enable GPU metric collection (default ``True``).
    kernels : bool
        Enable CUDA kernel tracing (default ``True``).
    memory : bool
        Enable GPU memory analysis (default ``True``).
    communication : bool
        Enable distributed communication profiling (default ``True``).
    data_loading : bool
        Enable data loading profiling (default ``True``).
    interval_ms : int
        Sampling interval in milliseconds for GPU and memory profilers
        (default 100).
    gpu_indices : list[int] | None
        Specific GPU indices to profile.  ``None`` means all GPUs.
    top_k_kernels : int
        Number of top kernels to surface (default 10).
    """

    def __init__(
        self,
        gpu: bool = True,
        kernels: bool = True,
        memory: bool = True,
        communication: bool = True,
        data_loading: bool = True,
        interval_ms: int = 100,
        gpu_indices: Optional[List[int]] = None,
        top_k_kernels: int = 10,
    ) -> None:
        self._enable_gpu = gpu
        self._enable_kernels = kernels
        self._enable_memory = memory
        self._enable_communication = communication
        self._enable_data_loading = data_loading
        self._interval_ms = interval_ms
        self._gpu_indices = gpu_indices
        self._top_k_kernels = top_k_kernels

        # Profiler instances (created on start)
        self._gpu_profiler: Optional[GPUProfiler] = None
        self._kernel_tracer: Optional[KernelTracer] = None
        self._memory_analyzer: Optional[MemoryAnalyzer] = None
        self._comm_profiler: Optional[CommunicationProfiler] = None
        self._data_profiler: Optional[DataLoadingProfiler] = None

        # Results (populated on stop)
        self._gpu_result: Optional[GPUProfileResult] = None
        self._kernel_result: Optional[KernelTraceResult] = None
        self._memory_result: Optional[MemoryAnalysisResult] = None
        self._comm_result: Optional[CommProfileResult] = None
        self._data_result: Optional[DataLoadingResult] = None
        self._summary: Optional[PerformanceSummary] = None

        self._active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all enabled profilers."""
        if self._active:
            logger.warning("ChamberProfiler is already active.")
            return

        self._clear_results()

        if self._enable_gpu:
            self._gpu_profiler = GPUProfiler(
                interval_ms=self._interval_ms,
                gpu_indices=self._gpu_indices,
            )
            self._gpu_profiler.start()

        if self._enable_kernels:
            self._kernel_tracer = KernelTracer(top_k=self._top_k_kernels)
            self._kernel_tracer.start()

        if self._enable_memory:
            self._memory_analyzer = MemoryAnalyzer(interval_ms=self._interval_ms)
            self._memory_analyzer.start()

        if self._enable_communication:
            self._comm_profiler = CommunicationProfiler()
            self._comm_profiler.start()

        if self._enable_data_loading:
            self._data_profiler = DataLoadingProfiler()
            self._data_profiler.start()

        self._active = True
        logger.debug("ChamberProfiler started.")

    def stop(self) -> PerformanceSummary:
        """Stop all profilers, run analysis, and return the summary."""
        if not self._active:
            logger.warning("ChamberProfiler is not active.")
            return self._empty_summary()

        self._active = False

        if self._gpu_profiler is not None:
            self._gpu_result = self._gpu_profiler.stop()

        if self._kernel_tracer is not None:
            self._kernel_result = self._kernel_tracer.stop()

        if self._memory_analyzer is not None:
            self._memory_result = self._memory_analyzer.stop()

        if self._comm_profiler is not None:
            self._comm_result = self._comm_profiler.stop()

        if self._data_profiler is not None:
            self._data_result = self._data_profiler.stop()

        self._summary = self._analyze()
        logger.debug("ChamberProfiler stopped.")
        return self._summary

    @contextmanager
    def profile(self) -> Iterator[ChamberProfiler]:
        """Context manager that profiles the enclosed block.

        Usage::

            profiler = ChamberProfiler()
            with profiler.profile():
                train()
            profiler.report()
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def report(
        self,
        format: str = "terminal",
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a report from the most recent profiling session.

        Parameters
        ----------
        format : str
            One of ``"terminal"``, ``"html"``, or ``"json"``.
        output_path : str | None
            File path for html/json output.

        Returns
        -------
        str | None
            Output file path for html/json, or ``None`` for terminal.
        """
        if self._summary is None:
            logger.warning("No profiling data. Call start()/stop() or use profile() first.")
            return None

        generator = ReportGenerator(
            performance_summary=self._summary,
            gpu_result=self._gpu_result,
            kernel_result=self._kernel_result,
            memory_result=self._memory_result,
            comm_result=self._comm_result,
            data_result=self._data_result,
        )
        return generator.generate_report(format=format, output_path=output_path)

    def get_summary(self) -> Optional[PerformanceSummary]:
        """Return the performance summary from the last profiling session."""
        return self._summary

    def get_results(self) -> Dict[str, Any]:
        """Return all raw profiler results as a dictionary."""
        return {
            "gpu": self._gpu_result,
            "kernels": self._kernel_result,
            "memory": self._memory_result,
            "communication": self._comm_result,
            "data_loading": self._data_result,
            "summary": self._summary,
        }

    def wrap_dataloader(self, dataloader: Any) -> Any:
        """Wrap a PyTorch DataLoader for profiling.

        Must be called after ``start()``.  Returns a profiled DataLoader
        wrapper that transparently measures batch load times.
        """
        if self._data_profiler is None:
            logger.warning("Data loading profiling is disabled.")
            return dataloader
        return self._data_profiler.wrap_dataloader(dataloader)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze(self) -> PerformanceSummary:
        """Run bottleneck detection on collected results."""
        detector = BottleneckDetector(
            gpu_result=self._gpu_result,
            kernel_result=self._kernel_result,
            memory_result=self._memory_result,
            comm_result=self._comm_result,
            data_result=self._data_result,
        )
        return detector.analyze()

    def _clear_results(self) -> None:
        """Reset all stored results."""
        self._gpu_result = None
        self._kernel_result = None
        self._memory_result = None
        self._comm_result = None
        self._data_result = None
        self._summary = None

    @staticmethod
    def _empty_summary() -> PerformanceSummary:
        """Return a minimal empty summary."""
        return PerformanceSummary(
            total_time_s=0.0,
            compute_pct=100.0,
            memory_pct=0.0,
            communication_pct=0.0,
            data_loading_pct=0.0,
            idle_pct=0.0,
            primary_bottleneck="compute",
            bottlenecks=[],
            overall_efficiency=0.0,
            summary_text="No profiling data available.",
        )
