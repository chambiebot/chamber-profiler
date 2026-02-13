"""Automatic profiler for ML training loops.

Hooks into a training loop, profiles the first N steps (default 100),
collects GPU, memory, kernel, and throughput metrics, and generates
actionable recommendations without user intervention.

Usage:
    profiler = AutoProfiler(profile_steps=100)
    for step, batch in enumerate(dataloader):
        with profiler.step(step):
            loss = model(batch)
            loss.backward()
            optimizer.step()
    report = profiler.report()
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Generator, List, Optional

from src.profiler.gpu_profiler import GPUProfiler, ProfileResult as GPUProfileResult, GPUMetricSnapshot

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_PROFILE_STEPS: int = 100
DEFAULT_WARMUP_STEPS: int = 5

# Thresholds for automatic recommendations
_LOW_UTILIZATION_PCT: float = 50.0
_HIGH_MEMORY_PCT: float = 90.0
_SLOW_STEP_RATIO: float = 2.0  # step > 2x median is "slow"


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class StepMetrics:
    """Metrics captured for a single training step."""

    step: int
    wall_time_s: float
    gpu_utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> StepMetrics:
        return cls(
            step=int(data.get("step", 0)),  # type: ignore[arg-type]
            wall_time_s=float(data.get("wall_time_s", 0.0)),  # type: ignore[arg-type]
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            memory_used_mb=float(data.get("memory_used_mb", 0.0)),  # type: ignore[arg-type]
            memory_total_mb=float(data.get("memory_total_mb", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class Recommendation:
    """A single actionable recommendation from auto-profiling."""

    category: str  # e.g. "gpu_utilization", "memory", "throughput"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    estimated_impact: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> Recommendation:
        return cls(
            category=str(data.get("category", "")),
            priority=str(data.get("priority", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            estimated_impact=data.get("estimated_impact"),  # type: ignore[arg-type]
        )


@dataclass
class AutoProfileResult:
    """Full result of an automatic profiling session."""

    total_steps_profiled: int
    warmup_steps: int
    step_metrics: List[StepMetrics]
    recommendations: List[Recommendation]
    avg_step_time_s: float = 0.0
    median_step_time_s: float = 0.0
    p95_step_time_s: float = 0.0
    avg_gpu_utilization_pct: float = 0.0
    peak_memory_mb: float = 0.0
    memory_total_mb: float = 0.0
    throughput_steps_per_sec: float = 0.0
    slow_steps: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_steps_profiled": self.total_steps_profiled,
            "warmup_steps": self.warmup_steps,
            "avg_step_time_s": self.avg_step_time_s,
            "median_step_time_s": self.median_step_time_s,
            "p95_step_time_s": self.p95_step_time_s,
            "avg_gpu_utilization_pct": self.avg_gpu_utilization_pct,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_total_mb": self.memory_total_mb,
            "throughput_steps_per_sec": self.throughput_steps_per_sec,
            "slow_steps": self.slow_steps,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "step_metrics": [s.to_dict() for s in self.step_metrics],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> AutoProfileResult:
        raw_metrics = data.get("step_metrics", [])
        metrics = [StepMetrics.from_dict(m) for m in raw_metrics]  # type: ignore[union-attr]
        raw_recs = data.get("recommendations", [])
        recs = [Recommendation.from_dict(r) for r in raw_recs]  # type: ignore[union-attr]
        return cls(
            total_steps_profiled=int(data.get("total_steps_profiled", 0)),  # type: ignore[arg-type]
            warmup_steps=int(data.get("warmup_steps", 0)),  # type: ignore[arg-type]
            step_metrics=metrics,
            recommendations=recs,
        )


# ============================================================================
# Auto Profiler
# ============================================================================


class AutoProfiler:
    """Automatic profiler that hooks into a training loop.

    Parameters
    ----------
    profile_steps:
        Number of steps to profile (after warmup).
    warmup_steps:
        Steps to skip before starting measurement. Warmup steps are
        still tracked but excluded from statistics.
    gpu_profiler:
        Optional GPUProfiler instance. If *None*, metrics are collected
        via wall-clock time only (no GPU telemetry).
    step_callback:
        Optional callable invoked with ``StepMetrics`` after each
        profiled step.
    """

    def __init__(
        self,
        profile_steps: int = DEFAULT_PROFILE_STEPS,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        gpu_profiler: Optional[GPUProfiler] = None,
        step_callback: Optional[Callable[[StepMetrics], None]] = None,
    ) -> None:
        self._profile_steps = profile_steps
        self._warmup_steps = warmup_steps
        self._gpu_profiler = gpu_profiler
        self._step_callback = step_callback

        self._metrics: List[StepMetrics] = []
        self._active = True
        self._total_steps_seen = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether the profiler is still collecting data."""
        return self._active

    @property
    def steps_collected(self) -> int:
        return len(self._metrics)

    @contextmanager
    def step(self, step_index: int) -> Generator[None, None, None]:
        """Context manager wrapping a single training step.

        After ``profile_steps + warmup_steps`` have been seen the
        profiler deactivates automatically.
        """
        self._total_steps_seen += 1
        total_needed = self._warmup_steps + self._profile_steps

        if not self._active or step_index >= total_needed:
            self._active = False
            yield
            return

        gpu_snap = self._collect_gpu_snapshot(step_index)

        t0 = time.monotonic()
        yield
        elapsed = time.monotonic() - t0

        util = gpu_snap.utilization_pct if gpu_snap else 0.0
        mem_used = gpu_snap.memory_used_mb if gpu_snap else 0.0
        mem_total = gpu_snap.memory_total_mb if gpu_snap else 0.0

        sm = StepMetrics(
            step=step_index,
            wall_time_s=elapsed,
            gpu_utilization_pct=util,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
        )
        self._metrics.append(sm)

        if self._step_callback is not None:
            self._step_callback(sm)

        if step_index >= total_needed - 1:
            self._active = False
            logger.info("AutoProfiler: profiling complete after step %d", step_index)

    def record_step(
        self,
        step_index: int,
        wall_time_s: float,
        gpu_utilization_pct: float = 0.0,
        memory_used_mb: float = 0.0,
        memory_total_mb: float = 0.0,
    ) -> None:
        """Manually record metrics for a step (non-context-manager API)."""
        total_needed = self._warmup_steps + self._profile_steps
        if not self._active or step_index >= total_needed:
            self._active = False
            return

        sm = StepMetrics(
            step=step_index,
            wall_time_s=wall_time_s,
            gpu_utilization_pct=gpu_utilization_pct,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
        )
        self._metrics.append(sm)
        if self._step_callback is not None:
            self._step_callback(sm)
        if step_index >= total_needed - 1:
            self._active = False

    def report(self) -> AutoProfileResult:
        """Compute statistics and generate recommendations.

        Only non-warmup steps are used for statistics. If no
        measurement steps were collected the result is empty.
        """
        measurement_metrics = [
            m for m in self._metrics if m.step >= self._warmup_steps
        ]

        if not measurement_metrics:
            return AutoProfileResult(
                total_steps_profiled=0,
                warmup_steps=self._warmup_steps,
                step_metrics=[],
                recommendations=[],
            )

        times = [m.wall_time_s for m in measurement_metrics]
        sorted_times = sorted(times)
        n = len(sorted_times)

        avg_time = sum(times) / n
        median_time = sorted_times[n // 2]
        p95_idx = min(int(n * 0.95), n - 1)
        p95_time = sorted_times[p95_idx]

        avg_util = sum(m.gpu_utilization_pct for m in measurement_metrics) / n
        peak_mem = max(m.memory_used_mb for m in measurement_metrics)
        mem_total = max(
            (m.memory_total_mb for m in measurement_metrics), default=0.0
        )

        throughput = 1.0 / avg_time if avg_time > 0 else 0.0

        # Detect slow steps (> 2x median)
        slow = [
            m.step
            for m in measurement_metrics
            if median_time > 0 and m.wall_time_s > _SLOW_STEP_RATIO * median_time
        ]

        recommendations = self._generate_recommendations(
            avg_util=avg_util,
            peak_mem=peak_mem,
            mem_total=mem_total,
            avg_time=avg_time,
            median_time=median_time,
            p95_time=p95_time,
            slow_steps=slow,
            measurement_metrics=measurement_metrics,
        )

        return AutoProfileResult(
            total_steps_profiled=n,
            warmup_steps=self._warmup_steps,
            step_metrics=self._metrics,
            recommendations=recommendations,
            avg_step_time_s=avg_time,
            median_step_time_s=median_time,
            p95_step_time_s=p95_time,
            avg_gpu_utilization_pct=avg_util,
            peak_memory_mb=peak_mem,
            memory_total_mb=mem_total,
            throughput_steps_per_sec=throughput,
            slow_steps=slow,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_gpu_snapshot(
        self, step_index: int
    ) -> Optional[GPUMetricSnapshot]:
        """Try to grab a single GPU snapshot. Returns None if unavailable."""
        if self._gpu_profiler is None:
            return None
        try:
            snap = self._gpu_profiler.snapshot()
            return snap if snap else None
        except Exception:
            logger.debug("Failed to collect GPU snapshot at step %d", step_index)
            return None

    def _generate_recommendations(
        self,
        avg_util: float,
        peak_mem: float,
        mem_total: float,
        avg_time: float,
        median_time: float,
        p95_time: float,
        slow_steps: List[int],
        measurement_metrics: List[StepMetrics],
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []

        # Low GPU utilization
        if avg_util > 0 and avg_util < _LOW_UTILIZATION_PCT:
            recs.append(
                Recommendation(
                    category="gpu_utilization",
                    priority="high",
                    title="Low GPU utilization detected",
                    description=(
                        f"Average GPU utilization is {avg_util:.1f}%, below "
                        f"the {_LOW_UTILIZATION_PCT:.0f}% threshold. Consider "
                        "increasing batch size, enabling mixed precision, or "
                        "using torch.compile() to increase GPU occupancy."
                    ),
                    estimated_impact=f"Could improve utilization by {_LOW_UTILIZATION_PCT - avg_util:.0f}+ percentage points",
                )
            )

        # High memory pressure
        if mem_total > 0:
            mem_pct = (peak_mem / mem_total) * 100
            if mem_pct > _HIGH_MEMORY_PCT:
                recs.append(
                    Recommendation(
                        category="memory",
                        priority="critical",
                        title="High GPU memory pressure",
                        description=(
                            f"Peak memory usage is {peak_mem:.0f} MB "
                            f"({mem_pct:.1f}% of {mem_total:.0f} MB). "
                            "Risk of OOM. Consider gradient checkpointing, "
                            "reducing batch size, or using mixed precision."
                        ),
                        estimated_impact=f"Could free ~{peak_mem - mem_total * 0.7:.0f} MB",
                    )
                )
            elif mem_pct < 50.0:
                recs.append(
                    Recommendation(
                        category="memory",
                        priority="medium",
                        title="GPU memory underutilized",
                        description=(
                            f"Peak memory usage is only {mem_pct:.1f}% of "
                            f"available {mem_total:.0f} MB. You could increase "
                            "batch size to improve throughput."
                        ),
                        estimated_impact="Potential 1.5-2x throughput increase",
                    )
                )

        # Step time variability
        if median_time > 0 and p95_time > 1.5 * median_time:
            recs.append(
                Recommendation(
                    category="throughput",
                    priority="high",
                    title="High step time variability",
                    description=(
                        f"P95 step time ({p95_time:.4f}s) is "
                        f"{p95_time / median_time:.1f}x the median "
                        f"({median_time:.4f}s). This suggests periodic "
                        "stalls — check data loading, garbage collection, "
                        "or communication overhead."
                    ),
                    estimated_impact=f"Reducing P95 to median could save {(p95_time - median_time) * 0.05 / avg_time * 100:.1f}% training time",
                )
            )

        # Slow steps
        if slow_steps:
            recs.append(
                Recommendation(
                    category="throughput",
                    priority="medium",
                    title=f"{len(slow_steps)} slow step(s) detected",
                    description=(
                        f"Steps {slow_steps[:5]} took >2x the median step "
                        "time. Investigate data loading stalls, host-device "
                        "sync, or periodic checkpointing overhead."
                    ),
                )
            )

        if not recs:
            recs.append(
                Recommendation(
                    category="general",
                    priority="low",
                    title="No major issues detected",
                    description=(
                        "Profiling looks healthy. GPU utilization, memory "
                        "usage, and step time variability are all within "
                        "acceptable ranges."
                    ),
                )
            )

        return recs
