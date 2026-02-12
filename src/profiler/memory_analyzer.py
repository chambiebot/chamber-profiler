"""GPU memory deep-dive module.

Tracks GPU memory allocation over time via a background sampling thread,
estimates memory breakdown by category (parameters, gradients, optimizer
state, activations), detects memory leaks, and assesses out-of-memory risk
with actionable suggestions.

Uses the PyTorch CUDA memory management APIs (``torch.cuda.memory_*``).
If PyTorch is not installed the analyzer logs a warning and produces empty
results -- the rest of the profiling suite keeps working.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch
# ---------------------------------------------------------------------------
_torch_available: bool = False
try:
    import torch  # type: ignore[import-untyped]

    _torch_available = True
    logger.debug("torch is available; GPU memory analysis is supported.")
except ImportError:
    logger.debug("torch is not installed; memory analysis will produce empty results.")


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class MemorySnapshot:
    """A single point-in-time measurement of GPU memory state."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    num_tensors: int
    device_index: int

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> MemorySnapshot:
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
            allocated_mb=float(data.get("allocated_mb", 0.0)),  # type: ignore[arg-type]
            reserved_mb=float(data.get("reserved_mb", 0.0)),  # type: ignore[arg-type]
            peak_allocated_mb=float(data.get("peak_allocated_mb", 0.0)),  # type: ignore[arg-type]
            peak_reserved_mb=float(data.get("peak_reserved_mb", 0.0)),  # type: ignore[arg-type]
            num_tensors=int(data.get("num_tensors", 0)),  # type: ignore[arg-type]
            device_index=int(data.get("device_index", 0)),  # type: ignore[arg-type]
        )


@dataclass
class MemoryBreakdown:
    """Estimated breakdown of GPU memory usage by category."""

    parameter_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    activation_memory_mb: float = 0.0
    other_mb: float = 0.0

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> MemoryBreakdown:
        return cls(
            parameter_memory_mb=float(data.get("parameter_memory_mb", 0.0)),  # type: ignore[arg-type]
            gradient_memory_mb=float(data.get("gradient_memory_mb", 0.0)),  # type: ignore[arg-type]
            optimizer_state_mb=float(data.get("optimizer_state_mb", 0.0)),  # type: ignore[arg-type]
            activation_memory_mb=float(data.get("activation_memory_mb", 0.0)),  # type: ignore[arg-type]
            other_mb=float(data.get("other_mb", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class OOMRiskAssessment:
    """Out-of-memory risk evaluation with actionable suggestions."""

    risk_score: float = 0.0
    peak_usage_pct: float = 0.0
    headroom_mb: float = 0.0
    suggestions: List[str] = field(default_factory=list)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "risk_score": self.risk_score,
            "peak_usage_pct": self.peak_usage_pct,
            "headroom_mb": self.headroom_mb,
            "suggestions": list(self.suggestions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> OOMRiskAssessment:
        raw_suggestions = data.get("suggestions", [])
        suggestions: List[str] = (
            [str(s) for s in raw_suggestions]  # type: ignore[union-attr]
            if isinstance(raw_suggestions, list)
            else []
        )
        return cls(
            risk_score=float(data.get("risk_score", 0.0)),  # type: ignore[arg-type]
            peak_usage_pct=float(data.get("peak_usage_pct", 0.0)),  # type: ignore[arg-type]
            headroom_mb=float(data.get("headroom_mb", 0.0)),  # type: ignore[arg-type]
            suggestions=suggestions,
        )


@dataclass
class MemoryAnalysisResult:
    """Aggregated output of a memory-analysis session."""

    snapshots: List[MemorySnapshot]
    breakdown: Optional[MemoryBreakdown] = None
    leak_detected: bool = False
    leak_points: List[Tuple[float, float]] = field(default_factory=list)
    oom_risk: OOMRiskAssessment = field(default_factory=OOMRiskAssessment)
    peak_snapshot: Optional[MemorySnapshot] = None

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "breakdown": self.breakdown.to_dict() if self.breakdown is not None else None,
            "leak_detected": self.leak_detected,
            "leak_points": [list(p) for p in self.leak_points],
            "oom_risk": self.oom_risk.to_dict(),
            "peak_snapshot": self.peak_snapshot.to_dict() if self.peak_snapshot is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> MemoryAnalysisResult:
        raw_snapshots = data.get("snapshots", [])
        snapshots = [
            MemorySnapshot.from_dict(s)  # type: ignore[arg-type]
            for s in raw_snapshots  # type: ignore[union-attr]
        ]

        raw_breakdown = data.get("breakdown")
        breakdown = (
            MemoryBreakdown.from_dict(raw_breakdown)  # type: ignore[arg-type]
            if raw_breakdown is not None
            else None
        )

        raw_leak_points = data.get("leak_points", [])
        leak_points: List[Tuple[float, float]] = [
            (float(p[0]), float(p[1]))  # type: ignore[index]
            for p in raw_leak_points  # type: ignore[union-attr]
        ]

        raw_oom = data.get("oom_risk")
        oom_risk = (
            OOMRiskAssessment.from_dict(raw_oom)  # type: ignore[arg-type]
            if raw_oom is not None
            else OOMRiskAssessment()
        )

        raw_peak = data.get("peak_snapshot")
        peak_snapshot = (
            MemorySnapshot.from_dict(raw_peak)  # type: ignore[arg-type]
            if raw_peak is not None
            else None
        )

        return cls(
            snapshots=snapshots,
            breakdown=breakdown,
            leak_detected=bool(data.get("leak_detected", False)),
            leak_points=leak_points,
            oom_risk=oom_risk,
            peak_snapshot=peak_snapshot,
        )


# ============================================================================
# Memory Analyzer
# ============================================================================


class MemoryAnalyzer:
    """Background-thread GPU memory tracker and analyzer.

    Usage::

        analyzer = MemoryAnalyzer(device_index=0)
        analyzer.start()
        # ... run workload ...
        result = analyzer.stop()
        print(result.oom_risk.risk_score, result.leak_detected)
    """

    # Number of consecutive increases required before flagging a leak.
    _LEAK_WINDOW_SIZE: int = 5

    # Minimum absolute growth (MB) over the leak window to flag.
    _LEAK_MIN_GROWTH_MB: float = 1.0

    def __init__(
        self,
        device_index: int = 0,
        track_tensors: bool = False,
        interval_ms: int = 100,
    ) -> None:
        self._device_index: int = device_index
        self._track_tensors: bool = track_tensors
        self._interval_s: float = interval_ms / 1000.0

        self._snapshots: List[MemorySnapshot] = []
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background memory sampling."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MemoryAnalyzer is already running; ignoring duplicate start().")
            return

        self._snapshots = []
        self._stop_event.clear()

        if not _torch_available:
            logger.warning(
                "torch is not installed. Memory analysis is disabled; "
                "stop() will return empty results."
            )

        if _torch_available and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device_index)

        self._thread = threading.Thread(
            target=self._collect_snapshots,
            name="memory-analyzer",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "Memory analyzer started (device=%d, interval=%.1f ms, track_tensors=%s).",
            self._device_index,
            self._interval_s * 1000,
            self._track_tensors,
        )

    def stop(self) -> MemoryAnalysisResult:
        """Stop sampling and return the full :class:`MemoryAnalysisResult`."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 5, 2.0))
            self._thread = None

        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            return self._empty_result()

        leak_points = self.detect_memory_leaks(snapshots)
        peak_snapshot = self.get_peak_analysis(snapshots)
        breakdown = self.get_memory_breakdown()
        oom_risk = self.assess_oom_risk(snapshots=snapshots, breakdown=breakdown)

        return MemoryAnalysisResult(
            snapshots=snapshots,
            breakdown=breakdown,
            leak_detected=len(leak_points) > 0,
            leak_points=leak_points,
            oom_risk=oom_risk,
            peak_snapshot=peak_snapshot,
        )

    def take_snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot right now.

        Returns a :class:`MemorySnapshot` regardless of whether the background
        sampler is running.  If CUDA is not available, returns a zeroed-out
        snapshot.
        """
        return self._read_memory_state()

    # ------------------------------------------------------------------
    # Background collection loop
    # ------------------------------------------------------------------

    def _collect_snapshots(self) -> None:
        """Run in the background thread, sampling memory at the configured interval."""
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            snapshot = self._read_memory_state()
            with self._lock:
                self._snapshots.append(snapshot)
            elapsed = time.monotonic() - loop_start
            sleep_time = self._interval_s - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    # ------------------------------------------------------------------
    # Memory reading
    # ------------------------------------------------------------------

    def _read_memory_state(self) -> MemorySnapshot:
        """Read current GPU memory state via torch.cuda APIs.

        Returns a zeroed-out :class:`MemorySnapshot` if CUDA is not available.
        """
        if not _torch_available or not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
                num_tensors=0,
                device_index=self._device_index,
            )

        device = self._device_index
        bytes_to_mb = 1.0 / (1024 * 1024)

        allocated_mb = torch.cuda.memory_allocated(device) * bytes_to_mb
        reserved_mb = torch.cuda.memory_reserved(device) * bytes_to_mb
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) * bytes_to_mb
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) * bytes_to_mb

        num_tensors = 0
        if self._track_tensors:
            num_tensors = self._count_cuda_tensors(device)

        return MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
            num_tensors=num_tensors,
            device_index=device,
        )

    @staticmethod
    def _count_cuda_tensors(device_index: int) -> int:
        """Count the number of live CUDA tensors on the given device.

        This triggers a GC collection first so that unreferenced tensors are
        cleaned up.  The count walks all objects tracked by the garbage
        collector, which can be slow for very large heaps -- callers should
        enable this only when needed.
        """
        gc.collect()
        count = 0
        try:
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        if obj.device.index == device_index:
                            count += 1
                except (ReferenceError, AttributeError):
                    # Weak references or objects being collected concurrently.
                    pass
        except Exception:
            logger.debug("Error while counting CUDA tensors.", exc_info=True)
        return count

    # ------------------------------------------------------------------
    # Memory breakdown
    # ------------------------------------------------------------------

    def get_memory_breakdown(
        self,
        model: Any = None,
        optimizer: Any = None,
    ) -> Optional[MemoryBreakdown]:
        """Estimate how GPU memory is distributed across categories.

        If *model* and/or *optimizer* are provided their parameter, gradient,
        and state sizes are computed directly.  Activation memory is estimated
        as the residual: ``total_allocated - params - grads - optimizer``.

        Returns ``None`` if CUDA is not available.
        """
        if not _torch_available or not torch.cuda.is_available():
            return None

        bytes_to_mb = 1.0 / (1024 * 1024)
        total_allocated_mb = torch.cuda.memory_allocated(self._device_index) * bytes_to_mb

        param_mb = 0.0
        grad_mb = 0.0
        optimizer_mb = 0.0

        if model is not None:
            param_mb, grad_mb = self._compute_model_memory(model)

        if optimizer is not None:
            optimizer_mb = self._compute_optimizer_memory(optimizer)

        known_mb = param_mb + grad_mb + optimizer_mb
        residual_mb = max(total_allocated_mb - known_mb, 0.0)

        # If model/optimizer were provided we attribute the residual to
        # activations.  Otherwise all allocated memory beyond known usage
        # is filed under "other".
        if model is not None:
            activation_mb = residual_mb
            other_mb = 0.0
        else:
            activation_mb = 0.0
            other_mb = residual_mb

        return MemoryBreakdown(
            parameter_memory_mb=round(param_mb, 3),
            gradient_memory_mb=round(grad_mb, 3),
            optimizer_state_mb=round(optimizer_mb, 3),
            activation_memory_mb=round(activation_mb, 3),
            other_mb=round(other_mb, 3),
        )

    @staticmethod
    def _compute_model_memory(model: Any) -> Tuple[float, float]:
        """Return (parameter_mb, gradient_mb) for a PyTorch model.

        Iterates over ``model.parameters()`` and sums element counts
        multiplied by element size.
        """
        bytes_to_mb = 1.0 / (1024 * 1024)
        param_bytes: float = 0.0
        grad_bytes: float = 0.0

        try:
            for param in model.parameters():
                param_bytes += param.nelement() * param.element_size()
                if param.grad is not None:
                    grad_bytes += param.grad.nelement() * param.grad.element_size()
        except Exception:
            logger.debug("Error computing model parameter memory.", exc_info=True)

        return param_bytes * bytes_to_mb, grad_bytes * bytes_to_mb

    @staticmethod
    def _compute_optimizer_memory(optimizer: Any) -> float:
        """Return total optimizer state memory in MB.

        Walks ``optimizer.state`` and sums the sizes of all tensor-valued
        state entries (e.g. momentum buffers, variance estimates in Adam).
        """
        bytes_to_mb = 1.0 / (1024 * 1024)
        state_bytes: float = 0.0

        try:
            for param_state in optimizer.state.values():
                if not isinstance(param_state, dict):
                    continue
                for value in param_state.values():
                    if isinstance(value, torch.Tensor):
                        state_bytes += value.nelement() * value.element_size()
        except Exception:
            logger.debug("Error computing optimizer state memory.", exc_info=True)

        return state_bytes * bytes_to_mb

    # ------------------------------------------------------------------
    # Leak detection
    # ------------------------------------------------------------------

    def detect_memory_leaks(
        self,
        snapshots: Optional[List[MemorySnapshot]] = None,
    ) -> List[Tuple[float, float]]:
        """Analyze snapshots for monotonically increasing allocation patterns.

        Looks for runs of at least :attr:`_LEAK_WINDOW_SIZE` consecutive
        snapshots where ``allocated_mb`` strictly increases, with cumulative
        growth exceeding :attr:`_LEAK_MIN_GROWTH_MB`.

        Parameters
        ----------
        snapshots:
            The snapshot series to analyze.  Defaults to the internally
            collected snapshots if not provided.

        Returns
        -------
        list[tuple[float, float]]
            Each entry is ``(timestamp, allocated_mb)`` marking a point where
            a leak was detected (the end of each monotonically increasing run).
        """
        if snapshots is None:
            with self._lock:
                snapshots = list(self._snapshots)

        if len(snapshots) < self._LEAK_WINDOW_SIZE:
            return []

        leak_points: List[Tuple[float, float]] = []
        run_start_idx = 0
        run_length = 1

        for i in range(1, len(snapshots)):
            if snapshots[i].allocated_mb > snapshots[i - 1].allocated_mb:
                run_length += 1
            else:
                # Check if the run that just ended qualifies as a leak.
                if run_length >= self._LEAK_WINDOW_SIZE:
                    run_end_idx = i - 1
                    growth = (
                        snapshots[run_end_idx].allocated_mb
                        - snapshots[run_start_idx].allocated_mb
                    )
                    if growth >= self._LEAK_MIN_GROWTH_MB:
                        leak_points.append(
                            (
                                snapshots[run_end_idx].timestamp,
                                snapshots[run_end_idx].allocated_mb,
                            )
                        )
                run_start_idx = i
                run_length = 1

        # Check the final run.
        if run_length >= self._LEAK_WINDOW_SIZE:
            run_end_idx = len(snapshots) - 1
            growth = (
                snapshots[run_end_idx].allocated_mb
                - snapshots[run_start_idx].allocated_mb
            )
            if growth >= self._LEAK_MIN_GROWTH_MB:
                leak_points.append(
                    (
                        snapshots[run_end_idx].timestamp,
                        snapshots[run_end_idx].allocated_mb,
                    )
                )

        return leak_points

    # ------------------------------------------------------------------
    # Peak analysis
    # ------------------------------------------------------------------

    def get_peak_analysis(
        self,
        snapshots: Optional[List[MemorySnapshot]] = None,
    ) -> Optional[MemorySnapshot]:
        """Identify the snapshot with the highest ``allocated_mb``.

        Parameters
        ----------
        snapshots:
            The snapshot series to analyze.  Defaults to the internally
            collected snapshots if not provided.

        Returns
        -------
        MemorySnapshot | None
            The snapshot at which peak allocation occurred, or ``None`` if
            no snapshots are available.
        """
        if snapshots is None:
            with self._lock:
                snapshots = list(self._snapshots)

        if not snapshots:
            return None

        return max(snapshots, key=lambda s: s.allocated_mb)

    # ------------------------------------------------------------------
    # OOM risk assessment
    # ------------------------------------------------------------------

    def assess_oom_risk(
        self,
        total_gpu_memory_mb: Optional[float] = None,
        snapshots: Optional[List[MemorySnapshot]] = None,
        breakdown: Optional[MemoryBreakdown] = None,
    ) -> OOMRiskAssessment:
        """Compute an out-of-memory risk score with actionable suggestions.

        Parameters
        ----------
        total_gpu_memory_mb:
            Total GPU memory in MB.  If ``None`` the value is queried from
            ``torch.cuda.get_device_properties``.
        snapshots:
            Snapshot series to base the assessment on.  Defaults to the
            internally collected snapshots.
        breakdown:
            Optional memory breakdown for more targeted suggestions.

        Returns
        -------
        OOMRiskAssessment
            Contains ``risk_score`` in [0, 1], ``peak_usage_pct``,
            ``headroom_mb``, and a list of ``suggestions``.
        """
        if snapshots is None:
            with self._lock:
                snapshots = list(self._snapshots)

        # Determine total GPU memory.
        if total_gpu_memory_mb is None:
            total_gpu_memory_mb = self._query_total_gpu_memory()

        if total_gpu_memory_mb is None or total_gpu_memory_mb <= 0:
            return OOMRiskAssessment(
                risk_score=0.0,
                peak_usage_pct=0.0,
                headroom_mb=0.0,
                suggestions=["Unable to determine total GPU memory."],
            )

        # Find peak allocation across all snapshots.
        peak_allocated_mb = 0.0
        if snapshots:
            peak_allocated_mb = max(s.allocated_mb for s in snapshots)
            # Also consider peak_allocated_mb reported by the allocator.
            peak_reported = max(s.peak_allocated_mb for s in snapshots)
            peak_allocated_mb = max(peak_allocated_mb, peak_reported)

        peak_usage_pct = (peak_allocated_mb / total_gpu_memory_mb) * 100.0
        headroom_mb = total_gpu_memory_mb - peak_allocated_mb

        # Risk score: linear ramp from 0 at 50% usage to 1 at 95% usage.
        if peak_usage_pct <= 50.0:
            risk_score = 0.0
        elif peak_usage_pct >= 95.0:
            risk_score = 1.0
        else:
            risk_score = (peak_usage_pct - 50.0) / (95.0 - 50.0)

        suggestions = self._generate_suggestions(peak_usage_pct, breakdown)

        return OOMRiskAssessment(
            risk_score=round(risk_score, 4),
            peak_usage_pct=round(peak_usage_pct, 2),
            headroom_mb=round(headroom_mb, 2),
            suggestions=suggestions,
        )

    def _query_total_gpu_memory(self) -> Optional[float]:
        """Query total GPU memory for the configured device via torch.cuda."""
        if not _torch_available or not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(self._device_index)
            return props.total_mem / (1024 * 1024)
        except Exception:
            logger.debug(
                "Failed to query GPU memory for device %d.", self._device_index,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Suggestion generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_suggestions(
        peak_pct: float,
        breakdown: Optional[MemoryBreakdown] = None,
    ) -> List[str]:
        """Generate actionable suggestions based on peak usage and breakdown.

        Parameters
        ----------
        peak_pct:
            Peak memory usage as a percentage of total GPU memory.
        breakdown:
            Optional memory breakdown; enables more targeted suggestions.

        Returns
        -------
        list[str]
            Human-readable suggestions.
        """
        suggestions: List[str] = []

        if peak_pct < 50.0:
            # Plenty of headroom -- no urgent suggestions.
            return suggestions

        # General suggestions ordered by typical impact.
        if peak_pct >= 70.0:
            suggestions.append(
                "Enable gradient checkpointing to trade compute for memory "
                "(e.g. model.gradient_checkpointing_enable())."
            )

        if peak_pct >= 60.0:
            suggestions.append(
                "Use mixed precision training (fp16/bf16) to halve activation and "
                "gradient memory (e.g. torch.cuda.amp.autocast)."
            )

        if peak_pct >= 80.0:
            suggestions.append(
                "Reduce batch size or use gradient accumulation to lower peak memory."
            )

        if peak_pct >= 90.0:
            suggestions.append(
                "Consider offloading optimizer states to CPU (e.g. DeepSpeed ZeRO-Offload)."
            )

        # Breakdown-specific suggestions.
        if breakdown is not None:
            total_known = (
                breakdown.parameter_memory_mb
                + breakdown.gradient_memory_mb
                + breakdown.optimizer_state_mb
                + breakdown.activation_memory_mb
                + breakdown.other_mb
            )

            if total_known > 0:
                # Large optimizer state relative to parameters suggests room for
                # memory-efficient optimizers.
                if (
                    breakdown.parameter_memory_mb > 0
                    and breakdown.optimizer_state_mb > 2.0 * breakdown.parameter_memory_mb
                ):
                    suggestions.append(
                        "Optimizer states are large relative to parameters. Consider "
                        "using a memory-efficient optimizer (e.g. 8-bit Adam via bitsandbytes, "
                        "or Adafactor)."
                    )

                # Activations dominating memory.
                if breakdown.activation_memory_mb > 0.5 * total_known:
                    suggestions.append(
                        "Activation memory dominates GPU usage. Enable gradient "
                        "checkpointing or reduce sequence length / input resolution."
                    )

        return suggestions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> MemoryAnalysisResult:
        """Return an empty :class:`MemoryAnalysisResult`."""
        return MemoryAnalysisResult(
            snapshots=[],
            breakdown=None,
            leak_detected=False,
            leak_points=[],
            oom_risk=OOMRiskAssessment(),
            peak_snapshot=None,
        )
