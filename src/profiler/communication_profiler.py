"""Distributed training communication analysis module.

Profiles inter-GPU collective communication operations (allreduce, allgather,
broadcast, reduce_scatter, etc.) during distributed training.  Hooks into
``torch.distributed`` collective APIs to record timing, data sizes, and
bandwidth for each operation, then analyses the results to surface bottleneck
ops, straggler ranks, and communication/compute overlap.

If PyTorch or ``torch.distributed`` is not available (or not initialised) the
profiler logs a warning and produces empty results -- the rest of the profiling
suite keeps working.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch and torch.distributed
# ---------------------------------------------------------------------------
_torch_available: bool = False
_dist_available: bool = False
try:
    import torch  # type: ignore[import-untyped]

    _torch_available = True
    logger.debug("torch is available; communication profiling may be supported.")
except ImportError:
    logger.debug("torch is not installed; communication profiling will produce empty results.")

if _torch_available:
    try:
        import torch.distributed as dist  # type: ignore[import-untyped]

        _dist_available = True
        logger.debug("torch.distributed is available.")
    except ImportError:
        logger.debug("torch.distributed is not available.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Collective operation names that we attempt to hook.
_COLLECTIVE_OPS: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce_scatter",
    "all_to_all",
    "reduce",
    "gather",
    "scatter",
    "barrier",
)

# Multiplier to convert bytes/microsecond to Gbps.
_BYTES_PER_US_TO_GBPS: float = 8.0 / 1000.0  # (8 bits/byte) / (1e6 us/s) * 1e9 = 8/1000


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CollectiveOperation:
    """A single recorded collective communication operation."""

    name: str
    duration_us: float
    data_size_bytes: int
    src_rank: Optional[int]
    dst_rank: Optional[int]
    timestamp: float
    bandwidth_gbps: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute bandwidth from data size and duration."""
        if self.duration_us > 0:
            bytes_per_us = self.data_size_bytes / self.duration_us
            self.bandwidth_gbps = round(bytes_per_us * _BYTES_PER_US_TO_GBPS, 4)
        else:
            self.bandwidth_gbps = 0.0

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CollectiveOperation:
        op = cls(
            name=str(data.get("name", "")),
            duration_us=float(data.get("duration_us", 0.0)),  # type: ignore[arg-type]
            data_size_bytes=int(data.get("data_size_bytes", 0)),  # type: ignore[arg-type]
            src_rank=_optional_int(data.get("src_rank")),
            dst_rank=_optional_int(data.get("dst_rank")),
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
        )
        # Allow overriding the computed bandwidth when deserialising.
        stored_bw = data.get("bandwidth_gbps")
        if stored_bw is not None:
            op.bandwidth_gbps = float(stored_bw)  # type: ignore[arg-type]
        return op


@dataclass
class CommProfileResult:
    """Aggregated output of a communication-profiling session."""

    operations: List[CollectiveOperation]
    total_comm_time_us: float
    total_data_transferred_bytes: int
    avg_bandwidth_gbps: float
    comm_compute_overlap_pct: float
    bottleneck_ops: List[CollectiveOperation]
    straggler_ranks: List[int]

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "total_comm_time_us": self.total_comm_time_us,
            "total_data_transferred_bytes": self.total_data_transferred_bytes,
            "avg_bandwidth_gbps": self.avg_bandwidth_gbps,
            "comm_compute_overlap_pct": self.comm_compute_overlap_pct,
            "bottleneck_ops": [op.to_dict() for op in self.bottleneck_ops],
            "straggler_ranks": list(self.straggler_ranks),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CommProfileResult:
        raw_ops = data.get("operations", [])
        operations = [
            CollectiveOperation.from_dict(op)  # type: ignore[arg-type]
            for op in raw_ops  # type: ignore[union-attr]
        ]

        raw_bottleneck = data.get("bottleneck_ops", [])
        bottleneck_ops = [
            CollectiveOperation.from_dict(op)  # type: ignore[arg-type]
            for op in raw_bottleneck  # type: ignore[union-attr]
        ]

        raw_stragglers = data.get("straggler_ranks", [])
        straggler_ranks: List[int] = (
            [int(r) for r in raw_stragglers]  # type: ignore[union-attr]
            if isinstance(raw_stragglers, list)
            else []
        )

        return cls(
            operations=operations,
            total_comm_time_us=float(data.get("total_comm_time_us", 0.0)),  # type: ignore[arg-type]
            total_data_transferred_bytes=int(data.get("total_data_transferred_bytes", 0)),  # type: ignore[arg-type]
            avg_bandwidth_gbps=float(data.get("avg_bandwidth_gbps", 0.0)),  # type: ignore[arg-type]
            comm_compute_overlap_pct=float(data.get("comm_compute_overlap_pct", 0.0)),  # type: ignore[arg-type]
            bottleneck_ops=bottleneck_ops,
            straggler_ranks=straggler_ranks,
        )


# ============================================================================
# Communication Profiler
# ============================================================================


class CommunicationProfiler:
    """Profile distributed training collective communication operations.

    Hooks into ``torch.distributed`` collective APIs to measure timing,
    data transfer sizes, and effective bandwidth.

    Usage::

        profiler = CommunicationProfiler()
        profiler.start()
        # ... run distributed training step(s) ...
        result = profiler.stop()
        for op in result.bottleneck_ops:
            print(op.name, op.duration_us, op.bandwidth_gbps)
    """

    # Bottleneck threshold: operations taking longer than this multiple of the
    # average duration are flagged.
    _BOTTLENECK_MULTIPLIER: float = 2.0

    # Straggler detection threshold: a rank whose total communication time
    # exceeds (mean + threshold * std) is considered a straggler.
    _STRAGGLER_SIGMA_THRESHOLD: float = 1.5

    def __init__(self) -> None:
        self._operations: List[CollectiveOperation] = []
        self._active: bool = False
        self._start_time: float = 0.0
        self._stop_time: float = 0.0

        # Compute events recorded while profiling is active.  Each entry is
        # (start_us, end_us) representing a contiguous compute interval.
        self._compute_events: List[Tuple[float, float]] = []

        # Original (unwrapped) functions we have monkey-patched.
        self._original_funcs: Dict[str, Callable[..., Any]] = {}

        # Whether hooks were successfully installed.
        self._hooks_installed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin profiling collective communication.

        Installs wrapper hooks on ``torch.distributed`` collective operations.
        If ``torch.distributed`` is not available or not initialised the call
        is a no-op and :meth:`stop` will return empty results.
        """
        if self._active:
            logger.warning(
                "CommunicationProfiler is already active; ignoring duplicate start()."
            )
            return

        self._operations = []
        self._compute_events = []
        self._original_funcs = {}
        self._hooks_installed = False
        self._start_time = time.monotonic()

        if not _torch_available:
            logger.warning(
                "torch is not installed. Communication profiling is disabled; "
                "stop() will return empty results."
            )
            self._active = True
            return

        if not _dist_available:
            logger.warning(
                "torch.distributed is not available. Communication profiling is "
                "disabled; stop() will return empty results."
            )
            self._active = True
            return

        if not dist.is_initialized():
            logger.warning(
                "torch.distributed is not initialised (no process group). "
                "Communication profiling is disabled; stop() will return empty results."
            )
            self._active = True
            return

        self._hook_nccl_ops()
        self._active = True
        logger.debug(
            "Communication profiler started (hooks_installed=%s).",
            self._hooks_installed,
        )

    def stop(self) -> CommProfileResult:
        """Stop profiling and return the analysed :class:`CommProfileResult`."""
        if not self._active:
            logger.warning(
                "CommunicationProfiler.stop() called but profiler is not active."
            )
            return self._empty_result()

        self._active = False
        self._stop_time = time.monotonic()

        # Restore original functions.
        self._unhook_nccl_ops()

        operations = list(self._operations)

        if not operations:
            return self._empty_result()

        total_comm_time = sum(op.duration_us for op in operations)
        total_data = sum(op.data_size_bytes for op in operations)

        bandwidths = [op.bandwidth_gbps for op in operations if op.bandwidth_gbps > 0]
        avg_bw = sum(bandwidths) / len(bandwidths) if bandwidths else 0.0

        overlap_pct = self.get_comm_compute_overlap()
        bottleneck = self._find_bottleneck_ops(operations)
        stragglers = self.detect_stragglers()

        return CommProfileResult(
            operations=operations,
            total_comm_time_us=round(total_comm_time, 2),
            total_data_transferred_bytes=total_data,
            avg_bandwidth_gbps=round(avg_bw, 4),
            comm_compute_overlap_pct=round(overlap_pct, 2),
            bottleneck_ops=bottleneck,
            straggler_ranks=stragglers,
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _hook_nccl_ops(self) -> None:
        """Install timing wrappers around ``torch.distributed`` collective ops.

        Each wrapper records the operation name, data size, and wall-clock
        duration, then calls the original function transparently.
        """
        if not _dist_available:
            return

        hooked_count = 0

        for op_name in _COLLECTIVE_OPS:
            original_fn = getattr(dist, op_name, None)
            if original_fn is None:
                logger.debug(
                    "torch.distributed.%s not found; skipping hook.", op_name
                )
                continue

            # Avoid double-wrapping if start() is called multiple times.
            if op_name in self._original_funcs:
                continue

            self._original_funcs[op_name] = original_fn
            wrapped = self._make_wrapper(op_name, original_fn)
            setattr(dist, op_name, wrapped)
            hooked_count += 1

        self._hooks_installed = hooked_count > 0
        logger.debug(
            "Hooked %d / %d collective operations.", hooked_count, len(_COLLECTIVE_OPS)
        )

    def _unhook_nccl_ops(self) -> None:
        """Restore original ``torch.distributed`` collective functions."""
        if not _dist_available:
            return

        for op_name, original_fn in self._original_funcs.items():
            try:
                setattr(dist, op_name, original_fn)
            except Exception:
                logger.debug(
                    "Failed to restore torch.distributed.%s.", op_name, exc_info=True
                )

        self._original_funcs = {}
        self._hooks_installed = False

    def _make_wrapper(
        self,
        op_name: str,
        original_fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Create a timing wrapper for a single collective operation.

        The wrapper measures wall-clock time, estimates the data size from the
        tensor arguments, and delegates to :meth:`_record_collective`.
        """
        profiler_self = self

        @functools.wraps(original_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not profiler_self._active:
                return original_fn(*args, **kwargs)

            data_size = _estimate_data_size(args, kwargs)
            src_rank, dst_rank = _extract_ranks(op_name, args, kwargs)

            # Synchronise CUDA before timing to get accurate wall-clock
            # measurements (collectives are async by default).
            if _torch_available and torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.monotonic()
            result = original_fn(*args, **kwargs)

            # Wait for the collective to complete (handle Work objects).
            if hasattr(result, "wait"):
                result.wait()

            if _torch_available and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.monotonic()

            profiler_self._record_collective(
                name=op_name,
                data_size=data_size,
                start_time=start_time,
                end_time=end_time,
                src_rank=src_rank,
                dst_rank=dst_rank,
            )

            return result

        return wrapper

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_collective(
        self,
        name: str,
        data_size: int,
        start_time: float,
        end_time: float,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
    ) -> None:
        """Record a collective operation.

        Parameters
        ----------
        name:
            Collective operation name (e.g. ``"all_reduce"``).
        data_size:
            Estimated data transfer size in bytes.
        start_time:
            ``time.monotonic()`` value at the start of the operation.
        end_time:
            ``time.monotonic()`` value at the end of the operation.
        src_rank:
            Source rank, if applicable (e.g. for broadcast).
        dst_rank:
            Destination rank, if applicable (e.g. for reduce).
        """
        duration_us = (end_time - start_time) * 1e6

        op = CollectiveOperation(
            name=name,
            duration_us=round(duration_us, 2),
            data_size_bytes=data_size,
            src_rank=src_rank,
            dst_rank=dst_rank,
            timestamp=start_time,
        )

        self._operations.append(op)
        logger.debug(
            "Recorded collective: %s  size=%d bytes  duration=%.1f us  bw=%.2f Gbps",
            name,
            data_size,
            duration_us,
            op.bandwidth_gbps,
        )

    # ------------------------------------------------------------------
    # Analysis: comm/compute overlap
    # ------------------------------------------------------------------

    def get_comm_compute_overlap(self) -> float:
        """Calculate the percentage of communication time that overlaps with compute.

        Overlap is estimated by checking how much of each collective operation's
        time interval intersects with recorded compute intervals.  If no compute
        events have been recorded we fall back to a heuristic based on whether
        the profiler observed any non-blocking (async) collectives that returned
        work handles.

        Returns
        -------
        float
            Overlap percentage in [0, 100].  Returns 0.0 when no operations
            were recorded.
        """
        if not self._operations:
            return 0.0

        if not self._compute_events:
            # Heuristic: in typical overlapping schedules (e.g. DDP with
            # gradient bucketing), communication is launched asynchronously
            # while the next backward pass or optimizer step runs.  Without
            # explicit compute events we cannot measure overlap, so report 0.
            return 0.0

        total_comm_us = 0.0
        overlapping_us = 0.0

        for op in self._operations:
            comm_start = op.timestamp
            comm_end = comm_start + op.duration_us / 1e6  # convert us back to s

            op_duration_us = op.duration_us
            total_comm_us += op_duration_us

            for comp_start, comp_end in self._compute_events:
                overlap_start = max(comm_start, comp_start)
                overlap_end = min(comm_end, comp_end)
                if overlap_end > overlap_start:
                    overlapping_us += (overlap_end - overlap_start) * 1e6

        if total_comm_us <= 0:
            return 0.0

        overlap_pct = min((overlapping_us / total_comm_us) * 100.0, 100.0)
        return round(overlap_pct, 2)

    def record_compute_event(self, start_time: float, end_time: float) -> None:
        """Record a compute interval for overlap analysis.

        Parameters
        ----------
        start_time:
            ``time.monotonic()`` value at the start of the compute interval.
        end_time:
            ``time.monotonic()`` value at the end of the compute interval.
        """
        if end_time > start_time:
            self._compute_events.append((start_time, end_time))

    # ------------------------------------------------------------------
    # Analysis: bandwidth utilisation
    # ------------------------------------------------------------------

    def get_bandwidth_utilization(
        self,
        theoretical_bw_gbps: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compare actual versus theoretical bandwidth.

        Parameters
        ----------
        theoretical_bw_gbps:
            Peak theoretical interconnect bandwidth in Gbps.  Common values:
            NVLink 3.0 = 600, NVLink 4.0 = 900, PCIe 4.0 x16 = 252,
            InfiniBand HDR = 200.  If ``None`` the method returns actual
            bandwidth statistics only (utilisation percentage will be 0).

        Returns
        -------
        dict[str, float]
            A dictionary with keys ``"avg_bandwidth_gbps"``,
            ``"peak_bandwidth_gbps"``, ``"theoretical_bw_gbps"``,
            and ``"utilization_pct"``.
        """
        if not self._operations:
            return {
                "avg_bandwidth_gbps": 0.0,
                "peak_bandwidth_gbps": 0.0,
                "theoretical_bw_gbps": theoretical_bw_gbps or 0.0,
                "utilization_pct": 0.0,
            }

        bandwidths = [op.bandwidth_gbps for op in self._operations if op.bandwidth_gbps > 0]

        avg_bw = sum(bandwidths) / len(bandwidths) if bandwidths else 0.0
        peak_bw = max(bandwidths) if bandwidths else 0.0

        utilization_pct = 0.0
        if theoretical_bw_gbps is not None and theoretical_bw_gbps > 0:
            utilization_pct = (avg_bw / theoretical_bw_gbps) * 100.0

        return {
            "avg_bandwidth_gbps": round(avg_bw, 4),
            "peak_bandwidth_gbps": round(peak_bw, 4),
            "theoretical_bw_gbps": round(theoretical_bw_gbps or 0.0, 4),
            "utilization_pct": round(utilization_pct, 2),
        }

    # ------------------------------------------------------------------
    # Analysis: straggler detection
    # ------------------------------------------------------------------

    def detect_stragglers(self) -> List[int]:
        """Identify GPU ranks that are significantly slower than their peers.

        Aggregates total communication time per source rank across all recorded
        operations.  A rank is considered a straggler if its total communication
        time exceeds ``mean + _STRAGGLER_SIGMA_THRESHOLD * std`` across all
        observed ranks.

        Returns
        -------
        list[int]
            Sorted list of straggler rank indices.  Returns an empty list if
            fewer than two distinct ranks were observed or if no operations
            have been recorded.
        """
        if not self._operations:
            return []

        # Aggregate total comm time per rank.
        rank_times: Dict[int, float] = {}
        for op in self._operations:
            rank = op.src_rank
            if rank is None:
                # For operations without an explicit source rank, try to infer
                # the local rank from torch.distributed.
                rank = self._get_local_rank()
            if rank is not None:
                rank_times[rank] = rank_times.get(rank, 0.0) + op.duration_us

        if len(rank_times) < 2:
            return []

        times = list(rank_times.values())
        mean_time = sum(times) / len(times)

        # Standard deviation.
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5

        if std_time <= 0:
            return []

        threshold = mean_time + self._STRAGGLER_SIGMA_THRESHOLD * std_time
        stragglers = sorted(
            rank for rank, t in rank_times.items() if t > threshold
        )
        return stragglers

    # ------------------------------------------------------------------
    # Bottleneck detection
    # ------------------------------------------------------------------

    def _find_bottleneck_ops(
        self, operations: List[CollectiveOperation]
    ) -> List[CollectiveOperation]:
        """Identify operations whose duration exceeds 2x the average.

        Parameters
        ----------
        operations:
            The list of recorded operations to analyse.

        Returns
        -------
        list[CollectiveOperation]
            Operations taking longer than ``_BOTTLENECK_MULTIPLIER * avg_duration``,
            sorted by duration descending.
        """
        if not operations:
            return []

        avg_duration = sum(op.duration_us for op in operations) / len(operations)
        threshold = avg_duration * self._BOTTLENECK_MULTIPLIER

        bottlenecks = [op for op in operations if op.duration_us > threshold]
        bottlenecks.sort(key=lambda op: op.duration_us, reverse=True)
        return bottlenecks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_local_rank() -> Optional[int]:
        """Return the local rank from ``torch.distributed`` if initialised."""
        if not _dist_available:
            return None
        try:
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            logger.debug("Failed to get local rank from torch.distributed.", exc_info=True)
        return None

    @staticmethod
    def _empty_result() -> CommProfileResult:
        """Return an empty :class:`CommProfileResult`."""
        return CommProfileResult(
            operations=[],
            total_comm_time_us=0.0,
            total_data_transferred_bytes=0,
            avg_bandwidth_gbps=0.0,
            comm_compute_overlap_pct=0.0,
            bottleneck_ops=[],
            straggler_ranks=[],
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _estimate_data_size(args: tuple, kwargs: dict) -> int:  # type: ignore[type-arg]
    """Estimate the total data size in bytes from collective op arguments.

    Scans both positional and keyword arguments for ``torch.Tensor`` objects
    and tensor lists, summing their ``nbytes`` (element_count * element_size).
    """
    total: int = 0

    for arg in args:
        total += _tensor_bytes(arg)

    for value in kwargs.values():
        total += _tensor_bytes(value)

    return total


def _tensor_bytes(obj: Any) -> int:
    """Return the byte size of a tensor or list of tensors, 0 otherwise."""
    if _torch_available and isinstance(obj, torch.Tensor):
        return obj.nelement() * obj.element_size()

    if isinstance(obj, (list, tuple)):
        total = 0
        for item in obj:
            if _torch_available and isinstance(item, torch.Tensor):
                total += item.nelement() * item.element_size()
        return total

    return 0


def _extract_ranks(
    op_name: str,
    args: tuple,  # type: ignore[type-arg]
    kwargs: dict,  # type: ignore[type-arg]
) -> Tuple[Optional[int], Optional[int]]:
    """Extract source and destination ranks from collective op arguments.

    Many collectives accept a ``src`` or ``dst`` keyword argument.  For
    broadcast the ``src`` parameter is typically the second positional arg
    (after the tensor).  This function makes a best-effort attempt to
    extract rank information.
    """
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None

    # Check keyword arguments first.
    if "src" in kwargs:
        src_rank = _safe_int(kwargs["src"])
    if "dst" in kwargs:
        dst_rank = _safe_int(kwargs["dst"])

    # For broadcast: broadcast(tensor, src, group=None, async_op=False)
    if op_name == "broadcast" and src_rank is None and len(args) >= 2:
        src_rank = _safe_int(args[1])

    # For reduce: reduce(tensor, dst, op=..., group=None, async_op=False)
    if op_name == "reduce" and dst_rank is None and len(args) >= 2:
        dst_rank = _safe_int(args[1])

    return src_rank, dst_rank


def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    """Convert a serialised value to ``Optional[int]``."""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
