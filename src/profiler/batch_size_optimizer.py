"""Batch size optimizer for ML training workloads.

Profiles training throughput at different batch sizes to find the sweet
spot — the largest batch size that maximizes throughput (samples/sec)
without hitting OOM or unacceptable memory pressure.

Works by running a configurable number of warm-up + measurement steps
at each candidate batch size and recording throughput, memory usage,
and GPU utilization.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch
# ---------------------------------------------------------------------------
_torch_available: bool = False
try:
    import torch  # type: ignore[import-untyped]
    _torch_available = True
except ImportError:
    pass


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class BatchSizeTrial:
    """Measurements from profiling at a single batch size."""

    batch_size: int
    throughput_samples_per_sec: float
    avg_step_time_ms: float
    peak_memory_mb: float
    avg_gpu_utilization_pct: float
    oom: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> BatchSizeTrial:
        return cls(
            batch_size=int(data.get("batch_size", 0)),  # type: ignore[arg-type]
            throughput_samples_per_sec=float(data.get("throughput_samples_per_sec", 0.0)),  # type: ignore[arg-type]
            avg_step_time_ms=float(data.get("avg_step_time_ms", 0.0)),  # type: ignore[arg-type]
            peak_memory_mb=float(data.get("peak_memory_mb", 0.0)),  # type: ignore[arg-type]
            avg_gpu_utilization_pct=float(data.get("avg_gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            oom=bool(data.get("oom", False)),
            error=data.get("error") if data.get("error") is not None else None,  # type: ignore[arg-type]
        )


@dataclass
class BatchSizeResult:
    """Full output of a batch size optimization analysis."""

    trials: List[BatchSizeTrial]
    optimal_batch_size: int
    optimal_throughput: float
    max_safe_batch_size: int
    memory_limit_mb: Optional[float]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "trials": [t.to_dict() for t in self.trials],
            "optimal_batch_size": self.optimal_batch_size,
            "optimal_throughput": self.optimal_throughput,
            "max_safe_batch_size": self.max_safe_batch_size,
            "memory_limit_mb": self.memory_limit_mb,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> BatchSizeResult:
        raw_trials = data.get("trials", [])
        trials = [
            BatchSizeTrial.from_dict(t)  # type: ignore[arg-type]
            for t in raw_trials  # type: ignore[union-attr]
        ] if isinstance(raw_trials, list) else []

        raw_recs = data.get("recommendations", [])
        recs = [str(r) for r in raw_recs] if isinstance(raw_recs, list) else []  # type: ignore[union-attr]

        mem = data.get("memory_limit_mb")

        return cls(
            trials=trials,
            optimal_batch_size=int(data.get("optimal_batch_size", 0)),  # type: ignore[arg-type]
            optimal_throughput=float(data.get("optimal_throughput", 0.0)),  # type: ignore[arg-type]
            max_safe_batch_size=int(data.get("max_safe_batch_size", 0)),  # type: ignore[arg-type]
            memory_limit_mb=float(mem) if mem is not None else None,  # type: ignore[arg-type]
            recommendations=recs,
        )


# ============================================================================
# Batch Size Optimizer
# ============================================================================


class BatchSizeOptimizer:
    """Profile training at different batch sizes to find the optimal one.

    Usage::

        optimizer = BatchSizeOptimizer(
            train_step_fn=train_one_step,
            batch_sizes=[8, 16, 32, 64, 128],
        )
        result = optimizer.run()
        print(f"Optimal batch size: {result.optimal_batch_size}")

    The ``train_step_fn`` callable receives the batch size as its only
    argument and should run one training step (forward + backward + optim),
    returning the number of samples processed (usually == batch_size).
    """

    def __init__(
        self,
        train_step_fn: Optional[Callable[[int], int]] = None,
        batch_sizes: Optional[List[int]] = None,
        warmup_steps: int = 3,
        measure_steps: int = 10,
        memory_limit_mb: Optional[float] = None,
        memory_headroom_pct: float = 10.0,
    ) -> None:
        self._train_step_fn = train_step_fn
        self._batch_sizes = batch_sizes or [8, 16, 32, 64, 128, 256]
        self._warmup_steps = max(warmup_steps, 1)
        self._measure_steps = max(measure_steps, 1)
        self._memory_limit_mb = memory_limit_mb
        self._memory_headroom_pct = memory_headroom_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BatchSizeResult:
        """Run batch size profiling across all candidate sizes.

        Returns
        -------
        BatchSizeResult
        """
        if self._train_step_fn is None:
            logger.warning("No train_step_fn provided; returning empty result.")
            return self._empty_result()

        trials: List[BatchSizeTrial] = []
        for bs in sorted(self._batch_sizes):
            trial = self._profile_batch_size(bs)
            trials.append(trial)
            if trial.oom:
                logger.info("OOM at batch_size=%d; skipping larger sizes.", bs)
                break

        return self._build_result(trials)

    def analyze_trials(self, trials: List[BatchSizeTrial]) -> BatchSizeResult:
        """Analyze pre-recorded trials without running profiling.

        Useful when trials were collected externally or loaded from disk.
        """
        return self._build_result(trials)

    def find_optimal(self, trials: List[BatchSizeTrial]) -> Tuple[int, float]:
        """Find the batch size with maximum throughput from completed trials.

        Returns (optimal_batch_size, optimal_throughput).
        """
        successful = [t for t in trials if not t.oom and t.error is None]
        if not successful:
            return 0, 0.0

        best = max(successful, key=lambda t: t.throughput_samples_per_sec)
        return best.batch_size, best.throughput_samples_per_sec

    def find_max_safe_batch_size(
        self, trials: List[BatchSizeTrial],
    ) -> int:
        """Find the largest batch size that didn't OOM and has headroom."""
        successful = [t for t in trials if not t.oom and t.error is None]
        if not successful:
            return 0

        mem_limit = self._memory_limit_mb
        if mem_limit is None:
            mem_limit = self._query_gpu_memory()

        if mem_limit is not None and mem_limit > 0:
            headroom = mem_limit * (self._memory_headroom_pct / 100.0)
            safe = [t for t in successful if t.peak_memory_mb < (mem_limit - headroom)]
            if safe:
                return max(t.batch_size for t in safe)

        # Fallback: just return the largest successful batch size.
        return max(t.batch_size for t in successful)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def _profile_batch_size(self, batch_size: int) -> BatchSizeTrial:
        """Profile a single batch size."""
        assert self._train_step_fn is not None

        # Clear GPU cache before each trial.
        if _torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup.
        try:
            for _ in range(self._warmup_steps):
                self._train_step_fn(batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._clear_oom()
                return BatchSizeTrial(
                    batch_size=batch_size,
                    throughput_samples_per_sec=0.0,
                    avg_step_time_ms=0.0,
                    peak_memory_mb=0.0,
                    avg_gpu_utilization_pct=0.0,
                    oom=True,
                    error=str(e),
                )
            raise

        # Reset memory stats after warmup.
        if _torch_available and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measurement.
        step_times: List[float] = []
        total_samples = 0
        try:
            for _ in range(self._measure_steps):
                start = time.perf_counter()
                samples = self._train_step_fn(batch_size)
                if _torch_available and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                step_times.append(elapsed)
                total_samples += samples
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._clear_oom()
                return BatchSizeTrial(
                    batch_size=batch_size,
                    throughput_samples_per_sec=0.0,
                    avg_step_time_ms=0.0,
                    peak_memory_mb=0.0,
                    avg_gpu_utilization_pct=0.0,
                    oom=True,
                    error=str(e),
                )
            raise

        total_time = sum(step_times)
        avg_step_ms = (total_time / len(step_times)) * 1000.0 if step_times else 0.0
        throughput = total_samples / total_time if total_time > 0 else 0.0

        peak_mem = 0.0
        if _torch_available and torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return BatchSizeTrial(
            batch_size=batch_size,
            throughput_samples_per_sec=round(throughput, 2),
            avg_step_time_ms=round(avg_step_ms, 2),
            peak_memory_mb=round(peak_mem, 2),
            avg_gpu_utilization_pct=0.0,  # would need GPU profiler for this
            oom=False,
        )

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, trials: List[BatchSizeTrial]) -> BatchSizeResult:
        """Build a BatchSizeResult from completed trials."""
        optimal_bs, optimal_tp = self.find_optimal(trials)
        max_safe = self.find_max_safe_batch_size(trials)
        recs = self._generate_recommendations(trials, optimal_bs, max_safe)

        return BatchSizeResult(
            trials=trials,
            optimal_batch_size=optimal_bs,
            optimal_throughput=round(optimal_tp, 2),
            max_safe_batch_size=max_safe,
            memory_limit_mb=self._memory_limit_mb,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(
        trials: List[BatchSizeTrial],
        optimal_bs: int,
        max_safe_bs: int,
    ) -> List[str]:
        """Generate recommendations from trial results."""
        recs: List[str] = []
        if not trials:
            return recs

        successful = [t for t in trials if not t.oom and t.error is None]
        oom_trials = [t for t in trials if t.oom]

        if optimal_bs > 0:
            recs.append(
                f"Optimal batch size is {optimal_bs} "
                f"({max(t.throughput_samples_per_sec for t in successful):.1f} samples/sec)."
            )

        if oom_trials:
            smallest_oom = min(t.batch_size for t in oom_trials)
            recs.append(
                f"OOM occurred at batch_size={smallest_oom}. "
                f"Maximum safe batch size is {max_safe_bs}."
            )

        if max_safe_bs > 0 and max_safe_bs != optimal_bs:
            recs.append(
                f"For maximum safety margin, use batch_size={max_safe_bs}. "
                f"For maximum throughput, use batch_size={optimal_bs}."
            )

        # Check for throughput plateau.
        if len(successful) >= 3:
            sorted_trials = sorted(successful, key=lambda t: t.batch_size)
            last_two_tp = [t.throughput_samples_per_sec for t in sorted_trials[-2:]]
            if len(last_two_tp) == 2 and last_two_tp[0] > 0:
                improvement = (last_two_tp[1] - last_two_tp[0]) / last_two_tp[0]
                if improvement < 0.05:
                    recs.append(
                        "Throughput is plateauing at larger batch sizes. "
                        "Increasing batch size further yields diminishing returns."
                    )

        # Gradient accumulation suggestion.
        if oom_trials and optimal_bs > 0:
            smallest_oom = min(t.batch_size for t in oom_trials)
            if smallest_oom > optimal_bs:
                accum_steps = smallest_oom // optimal_bs
                recs.append(
                    f"To simulate batch_size={smallest_oom} without OOM, use "
                    f"gradient accumulation with {accum_steps} steps of "
                    f"batch_size={optimal_bs}."
                )

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_gpu_memory() -> Optional[float]:
        """Query total GPU memory in MB."""
        if not _torch_available or not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_mem / (1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def _clear_oom() -> None:
        """Attempt to recover from OOM by clearing cache."""
        if _torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _empty_result() -> BatchSizeResult:
        return BatchSizeResult(
            trials=[],
            optimal_batch_size=0,
            optimal_throughput=0.0,
            max_safe_batch_size=0,
            memory_limit_mb=None,
            recommendations=[],
        )
