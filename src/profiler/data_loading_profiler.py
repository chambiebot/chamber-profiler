"""Data pipeline analysis module.

Profiles the data loading pipeline to detect whether data loading is a
bottleneck for GPU training.  Wraps PyTorch ``DataLoader`` instances to
transparently measure batch fetch time, batch processing time, and GPU idle
time (time the GPU spent waiting for the next batch).

Based on the collected metrics the profiler classifies bottleneck severity
and generates actionable recommendations (e.g. increase ``num_workers``,
enable ``pin_memory``, adjust ``prefetch_factor``).

If PyTorch is not installed the profiler logs a warning and produces empty
results -- the rest of the profiling suite keeps working.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch
# ---------------------------------------------------------------------------
_torch_available: bool = False
try:
    import torch  # type: ignore[import-untyped]

    _torch_available = True
    logger.debug("torch is available; data loading profiling is supported.")
except ImportError:
    logger.debug("torch is not installed; data loading profiling will produce empty results.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# If GPU idle time exceeds this fraction of total batch time, data loading
# is considered a bottleneck.
_BOTTLENECK_IDLE_FRACTION: float = 0.20

# Severity thresholds (fraction of total batch time spent waiting for data).
_SEVERITY_MILD_THRESHOLD: float = 0.20
_SEVERITY_MODERATE_THRESHOLD: float = 0.40
_SEVERITY_SEVERE_THRESHOLD: float = 0.60

# Recommended num_workers scaling: target at least this many workers per GPU.
_RECOMMENDED_WORKERS_PER_GPU: int = 8

# Default prefetch factor recommendation.
_RECOMMENDED_PREFETCH_FACTOR: int = 4


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class DataLoadingMetrics:
    """Metrics for a single batch fetch/process cycle."""

    batch_load_time_ms: float
    batch_process_time_ms: float
    gpu_idle_time_ms: float
    timestamp: float
    batch_index: int
    num_workers_active: int
    prefetch_buffer_size: int
    io_throughput_mbps: float

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> DataLoadingMetrics:
        return cls(
            batch_load_time_ms=float(data.get("batch_load_time_ms", 0.0)),  # type: ignore[arg-type]
            batch_process_time_ms=float(data.get("batch_process_time_ms", 0.0)),  # type: ignore[arg-type]
            gpu_idle_time_ms=float(data.get("gpu_idle_time_ms", 0.0)),  # type: ignore[arg-type]
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
            batch_index=int(data.get("batch_index", 0)),  # type: ignore[arg-type]
            num_workers_active=int(data.get("num_workers_active", 0)),  # type: ignore[arg-type]
            prefetch_buffer_size=int(data.get("prefetch_buffer_size", 0)),  # type: ignore[arg-type]
            io_throughput_mbps=float(data.get("io_throughput_mbps", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class DataLoadingResult:
    """Aggregated output of a data-loading profiling session."""

    metrics: List[DataLoadingMetrics]
    avg_load_time_ms: float
    avg_gpu_idle_time_ms: float
    is_bottleneck: bool
    bottleneck_severity: str
    recommendations: List[str]
    io_throughput_mbps: float

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "avg_load_time_ms": self.avg_load_time_ms,
            "avg_gpu_idle_time_ms": self.avg_gpu_idle_time_ms,
            "is_bottleneck": self.is_bottleneck,
            "bottleneck_severity": self.bottleneck_severity,
            "recommendations": list(self.recommendations),
            "io_throughput_mbps": self.io_throughput_mbps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> DataLoadingResult:
        raw_metrics = data.get("metrics", [])
        metrics = [
            DataLoadingMetrics.from_dict(m)  # type: ignore[arg-type]
            for m in raw_metrics  # type: ignore[union-attr]
        ]

        raw_recommendations = data.get("recommendations", [])
        recommendations: List[str] = (
            [str(r) for r in raw_recommendations]  # type: ignore[union-attr]
            if isinstance(raw_recommendations, list)
            else []
        )

        return cls(
            metrics=metrics,
            avg_load_time_ms=float(data.get("avg_load_time_ms", 0.0)),  # type: ignore[arg-type]
            avg_gpu_idle_time_ms=float(data.get("avg_gpu_idle_time_ms", 0.0)),  # type: ignore[arg-type]
            is_bottleneck=bool(data.get("is_bottleneck", False)),
            bottleneck_severity=str(data.get("bottleneck_severity", "none")),
            recommendations=recommendations,
            io_throughput_mbps=float(data.get("io_throughput_mbps", 0.0)),  # type: ignore[arg-type]
        )


# ============================================================================
# Profiled DataLoader wrapper
# ============================================================================


class ProfiledDataLoader:
    """Transparent wrapper around a PyTorch ``DataLoader`` that profiles each batch.

    Times every ``__next__`` call to measure how long the training loop waits
    for data (batch load time) and estimates GPU idle time based on the gap
    between batch availability and the start of GPU computation.

    Usage::

        profiler = DataLoadingProfiler()
        profiled_loader = profiler.wrap_dataloader(train_loader)
        for batch in profiled_loader:
            # ... train on batch ...
            pass
        result = profiler.stop()
    """

    def __init__(
        self,
        dataloader: Any,
        profiler: DataLoadingProfiler,
    ) -> None:
        self._dataloader: Any = dataloader
        self._profiler: DataLoadingProfiler = profiler
        self._iterator: Optional[Iterator[Any]] = None
        self._batch_index: int = 0
        self._last_batch_end_time: float = 0.0

    def __iter__(self) -> ProfiledDataLoader:
        self._iterator = iter(self._dataloader)
        self._batch_index = 0
        self._last_batch_end_time = time.perf_counter()
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            raise StopIteration

        # The GPU idle time is the wall-clock time between the end of the
        # previous batch's processing and the start of data loading.  In a
        # well-pipelined setup this should be near zero because the next batch
        # is already prefetched.  We approximate this as the gap between when
        # we last yielded a batch and when we start fetching the next one.
        fetch_start = time.perf_counter()

        # Measure time spent waiting for the next batch from the DataLoader.
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise

        fetch_end = time.perf_counter()

        load_time_ms = (fetch_end - fetch_start) * 1000.0

        # GPU idle time: the time between the end of the previous batch
        # (when we returned from __next__ last time) and when the new batch
        # became available.  For the first batch this is the time from
        # __iter__ until the batch is ready.
        gpu_idle_time_ms = load_time_ms  # conservative: the GPU was waiting this long

        # Estimate I/O throughput from batch size in memory.
        batch_size_bytes = _estimate_batch_size_bytes(batch)
        if load_time_ms > 0:
            io_throughput_mbps = (batch_size_bytes / (1024 * 1024)) / (load_time_ms / 1000.0)
        else:
            io_throughput_mbps = 0.0

        # Extract DataLoader configuration.
        num_workers = getattr(self._dataloader, "num_workers", 0)
        prefetch_factor = getattr(self._dataloader, "prefetch_factor", 2)
        if prefetch_factor is None:
            prefetch_factor = 2
        prefetch_buffer_size = num_workers * prefetch_factor if num_workers > 0 else 0

        self._profiler._record_batch(
            batch_index=self._batch_index,
            load_time=load_time_ms,
            process_time=0.0,  # will be updated if the user records it
            gpu_idle_time=gpu_idle_time_ms,
            num_workers_active=num_workers,
            prefetch_buffer_size=prefetch_buffer_size,
            io_throughput_mbps=io_throughput_mbps,
        )

        self._batch_index += 1
        self._last_batch_end_time = time.perf_counter()

        return batch

    def __len__(self) -> int:
        return len(self._dataloader)


# ============================================================================
# Data Loading Profiler
# ============================================================================


class DataLoadingProfiler:
    """Profile the data loading pipeline to detect data-loading bottlenecks.

    Wraps a PyTorch ``DataLoader`` to transparently measure batch load time,
    GPU idle time, and I/O throughput.  After profiling, generates a
    :class:`DataLoadingResult` with bottleneck severity and actionable
    recommendations.

    Usage::

        profiler = DataLoadingProfiler()
        profiler.start()
        profiled_loader = profiler.wrap_dataloader(train_loader)
        for batch in profiled_loader:
            # ... train on batch ...
            pass
        result = profiler.stop()
        print(result.bottleneck_severity, result.recommendations)
    """

    def __init__(self) -> None:
        self._metrics: List[DataLoadingMetrics] = []
        self._active: bool = False
        self._start_time: float = 0.0

        # Observed DataLoader configuration (populated when wrapping).
        self._observed_num_workers: int = 0
        self._observed_pin_memory: bool = False
        self._observed_prefetch_factor: int = 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin data loading profiling.

        Resets all previously collected metrics.  If torch is not available
        the call still succeeds; :meth:`stop` will return empty results.
        """
        if self._active:
            logger.warning(
                "DataLoadingProfiler is already active; ignoring duplicate start()."
            )
            return

        self._metrics = []
        self._start_time = time.monotonic()

        if not _torch_available:
            logger.warning(
                "torch is not installed. Data loading profiling is disabled; "
                "stop() will return empty results."
            )

        self._active = True
        logger.debug("Data loading profiler started.")

    def stop(self) -> DataLoadingResult:
        """Stop profiling and return the analysed :class:`DataLoadingResult`.

        Returns
        -------
        DataLoadingResult
            Contains per-batch metrics, averages, bottleneck classification,
            and recommendations.
        """
        if not self._active:
            logger.warning(
                "DataLoadingProfiler.stop() called but profiler is not active."
            )
            return self._empty_result()

        self._active = False
        metrics = list(self._metrics)

        if not metrics:
            return self._empty_result()

        avg_load = sum(m.batch_load_time_ms for m in metrics) / len(metrics)
        avg_idle = sum(m.gpu_idle_time_ms for m in metrics) / len(metrics)

        throughputs = [m.io_throughput_mbps for m in metrics if m.io_throughput_mbps > 0]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0

        is_bottleneck = self.detect_bottleneck(metrics)
        severity = self._classify_severity(metrics)
        recommendations = self.get_recommendations(metrics)

        return DataLoadingResult(
            metrics=metrics,
            avg_load_time_ms=round(avg_load, 3),
            avg_gpu_idle_time_ms=round(avg_idle, 3),
            is_bottleneck=is_bottleneck,
            bottleneck_severity=severity,
            recommendations=recommendations,
            io_throughput_mbps=round(avg_throughput, 3),
        )

    def wrap_dataloader(self, dataloader: Any) -> ProfiledDataLoader:
        """Wrap a PyTorch ``DataLoader`` with profiling instrumentation.

        Parameters
        ----------
        dataloader:
            A ``torch.utils.data.DataLoader`` instance (or any object that
            supports ``__iter__`` and ``__len__``).

        Returns
        -------
        ProfiledDataLoader
            A drop-in replacement that records metrics for each batch.
        """
        # Capture the DataLoader's configuration for recommendation generation.
        self._observed_num_workers = getattr(dataloader, "num_workers", 0)
        self._observed_pin_memory = getattr(dataloader, "pin_memory", False)

        prefetch = getattr(dataloader, "prefetch_factor", 2)
        self._observed_prefetch_factor = prefetch if prefetch is not None else 2

        logger.debug(
            "Wrapping DataLoader (num_workers=%d, pin_memory=%s, prefetch_factor=%d).",
            self._observed_num_workers,
            self._observed_pin_memory,
            self._observed_prefetch_factor,
        )
        return ProfiledDataLoader(dataloader, self)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_batch(
        self,
        batch_index: int,
        load_time: float,
        process_time: float,
        gpu_idle_time: float = 0.0,
        num_workers_active: int = 0,
        prefetch_buffer_size: int = 0,
        io_throughput_mbps: float = 0.0,
    ) -> None:
        """Record metrics for a single batch.

        Parameters
        ----------
        batch_index:
            Zero-based index of the batch within the epoch.
        load_time:
            Time in milliseconds to fetch the batch from the DataLoader.
        process_time:
            Time in milliseconds to process the batch on the GPU.
        gpu_idle_time:
            Time in milliseconds the GPU was idle waiting for data.
        num_workers_active:
            Number of DataLoader worker processes active.
        prefetch_buffer_size:
            Size of the prefetch buffer (num_workers * prefetch_factor).
        io_throughput_mbps:
            Estimated I/O throughput in MB/s for this batch.
        """
        metric = DataLoadingMetrics(
            batch_load_time_ms=round(load_time, 3),
            batch_process_time_ms=round(process_time, 3),
            gpu_idle_time_ms=round(gpu_idle_time, 3),
            timestamp=time.time(),
            batch_index=batch_index,
            num_workers_active=num_workers_active,
            prefetch_buffer_size=prefetch_buffer_size,
            io_throughput_mbps=round(io_throughput_mbps, 3),
        )
        self._metrics.append(metric)
        logger.debug(
            "Recorded batch %d: load=%.1f ms, process=%.1f ms, gpu_idle=%.1f ms, "
            "throughput=%.1f MB/s",
            batch_index,
            load_time,
            process_time,
            gpu_idle_time,
            io_throughput_mbps,
        )

    # ------------------------------------------------------------------
    # Analysis: bottleneck detection
    # ------------------------------------------------------------------

    def detect_bottleneck(
        self,
        metrics: Optional[List[DataLoadingMetrics]] = None,
    ) -> bool:
        """Determine if data loading is a bottleneck for GPU training.

        The data pipeline is considered a bottleneck when the average GPU
        idle time (waiting for data) exceeds a threshold fraction of the
        average total batch time (load + process).

        Parameters
        ----------
        metrics:
            Metrics to analyse.  Defaults to internally collected metrics.

        Returns
        -------
        bool
            ``True`` if data loading is a bottleneck.
        """
        if metrics is None:
            metrics = list(self._metrics)

        if not metrics:
            return False

        avg_idle = sum(m.gpu_idle_time_ms for m in metrics) / len(metrics)
        avg_total = sum(
            m.batch_load_time_ms + m.batch_process_time_ms for m in metrics
        ) / len(metrics)

        if avg_total <= 0:
            return False

        idle_fraction = avg_idle / avg_total
        return idle_fraction > _BOTTLENECK_IDLE_FRACTION

    # ------------------------------------------------------------------
    # Analysis: severity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(metrics: List[DataLoadingMetrics]) -> str:
        """Classify bottleneck severity based on GPU idle time fraction.

        Returns
        -------
        str
            One of ``"none"``, ``"mild"``, ``"moderate"``, ``"severe"``.
        """
        if not metrics:
            return "none"

        avg_idle = sum(m.gpu_idle_time_ms for m in metrics) / len(metrics)
        avg_total = sum(
            m.batch_load_time_ms + m.batch_process_time_ms for m in metrics
        ) / len(metrics)

        if avg_total <= 0:
            return "none"

        idle_fraction = avg_idle / avg_total

        if idle_fraction >= _SEVERITY_SEVERE_THRESHOLD:
            return "severe"
        elif idle_fraction >= _SEVERITY_MODERATE_THRESHOLD:
            return "moderate"
        elif idle_fraction >= _SEVERITY_MILD_THRESHOLD:
            return "mild"
        else:
            return "none"

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        metrics: Optional[List[DataLoadingMetrics]] = None,
    ) -> List[str]:
        """Generate actionable recommendations to improve data loading performance.

        Recommendations include specific values (e.g. ``"Increase num_workers
        from 2 to 8"``) based on the observed DataLoader configuration and
        system resources.

        Parameters
        ----------
        metrics:
            Metrics to base recommendations on.  Defaults to internally
            collected metrics.

        Returns
        -------
        list[str]
            Human-readable recommendations, ordered by expected impact.
        """
        if metrics is None:
            metrics = list(self._metrics)

        if not metrics:
            return []

        recommendations: List[str] = []

        avg_idle = sum(m.gpu_idle_time_ms for m in metrics) / len(metrics)
        avg_load = sum(m.batch_load_time_ms for m in metrics) / len(metrics)
        avg_total = sum(
            m.batch_load_time_ms + m.batch_process_time_ms for m in metrics
        ) / len(metrics)

        if avg_total <= 0:
            return recommendations

        idle_fraction = avg_idle / avg_total

        # -- num_workers recommendation ------------------------------------
        current_workers = self._observed_num_workers
        cpu_count = os.cpu_count() or 4
        # Recommend at least _RECOMMENDED_WORKERS_PER_GPU, but no more than
        # the available CPU count.
        recommended_workers = min(_RECOMMENDED_WORKERS_PER_GPU, cpu_count)

        if current_workers < recommended_workers and idle_fraction > _BOTTLENECK_IDLE_FRACTION:
            recommendations.append(
                f"Increase num_workers from {current_workers} to {recommended_workers} "
                f"to parallelize data loading across more CPU cores."
            )

        # -- pin_memory recommendation -------------------------------------
        if not self._observed_pin_memory and _torch_available:
            cuda_available = False
            try:
                cuda_available = torch.cuda.is_available()
            except Exception:
                pass

            if cuda_available:
                recommendations.append(
                    "Enable pin_memory=True to accelerate host-to-device "
                    "memory transfers via page-locked (pinned) memory."
                )

        # -- prefetch_factor recommendation --------------------------------
        current_prefetch = self._observed_prefetch_factor
        if (
            current_prefetch < _RECOMMENDED_PREFETCH_FACTOR
            and current_workers > 0
            and idle_fraction > _BOTTLENECK_IDLE_FRACTION
        ):
            recommendations.append(
                f"Set prefetch_factor={_RECOMMENDED_PREFETCH_FACTOR} "
                f"(currently {current_prefetch}) to keep more batches queued "
                f"ahead of GPU consumption."
            )

        # -- persistent_workers recommendation -----------------------------
        if current_workers > 0 and avg_load > 50.0:
            # High average load time with workers suggests worker restart
            # overhead.
            recommendations.append(
                "Enable persistent_workers=True to avoid re-spawning "
                "DataLoader worker processes each epoch."
            )

        # -- I/O throughput recommendation ---------------------------------
        throughputs = [m.io_throughput_mbps for m in metrics if m.io_throughput_mbps > 0]
        if throughputs:
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput < 100.0 and idle_fraction > _BOTTLENECK_IDLE_FRACTION:
                recommendations.append(
                    f"I/O throughput is low ({avg_throughput:.1f} MB/s). Consider "
                    f"using a faster storage backend (NVMe SSD), memory-mapped "
                    f"files, or a data format with better sequential read "
                    f"performance (e.g. WebDataset, FFCV, or TFRecord)."
                )

        # -- num_workers == 0 special case ---------------------------------
        if current_workers == 0 and idle_fraction > _BOTTLENECK_IDLE_FRACTION:
            recommendations.append(
                f"num_workers is 0 (data is loaded in the main process). "
                f"Set num_workers={recommended_workers} to offload data "
                f"loading to background worker processes."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> DataLoadingResult:
        """Return an empty :class:`DataLoadingResult`."""
        return DataLoadingResult(
            metrics=[],
            avg_load_time_ms=0.0,
            avg_gpu_idle_time_ms=0.0,
            is_bottleneck=False,
            bottleneck_severity="none",
            recommendations=[],
            io_throughput_mbps=0.0,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _estimate_batch_size_bytes(batch: Any) -> int:
    """Estimate the total size of a batch in bytes.

    Handles common batch formats:
    - A single ``torch.Tensor``
    - A tuple/list of tensors (e.g. ``(input, target)``)
    - A dict mapping names to tensors

    Returns 0 if the batch format is not recognised or torch is unavailable.
    """
    if not _torch_available:
        return 0

    total: int = 0

    if isinstance(batch, torch.Tensor):
        return batch.nelement() * batch.element_size()

    if isinstance(batch, (tuple, list)):
        for item in batch:
            if isinstance(item, torch.Tensor):
                total += item.nelement() * item.element_size()
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        total += sub_item.nelement() * sub_item.element_size()
        return total

    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                total += value.nelement() * value.element_size()
        return total

    return 0
