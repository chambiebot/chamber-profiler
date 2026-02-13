"""ROI calculator for profiler recommendations.

Given a set of optimization recommendations (e.g. increase batch size
from 32 to 64), compute the expected return on investment in terms of
wall-clock time saved, GPU-hours freed, and dollar cost reduction.

Designed to integrate with the optimization_advisor and cost_analyzer
outputs to give a single "is this worth doing?" answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# GPU pricing (simplified, re-used from cost_analyzer)
# ============================================================================

GPU_HOURLY_COST: Dict[str, float] = {
    "a100_40gb": 3.40,
    "a100_80gb": 4.10,
    "h100_80gb": 5.50,
    "a10g": 1.20,
    "l4": 0.81,
    "v100": 1.46,
    "t4": 0.53,
}


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class OptimizationScenario:
    """A single optimization to evaluate.

    Describe the before/after state of one parameter change.
    """

    name: str
    category: str  # e.g. "batch_size", "precision", "data_loading"
    before_value: str  # human-readable, e.g. "batch_size=32"
    after_value: str  # e.g. "batch_size=64"
    before_step_time_s: float
    after_step_time_s: float  # estimated
    before_memory_mb: float = 0.0
    after_memory_mb: float = 0.0
    confidence: float = 0.8  # 0-1, how confident the estimate is

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> OptimizationScenario:
        return cls(
            name=str(data.get("name", "")),
            category=str(data.get("category", "")),
            before_value=str(data.get("before_value", "")),
            after_value=str(data.get("after_value", "")),
            before_step_time_s=float(data.get("before_step_time_s", 0.0)),  # type: ignore[arg-type]
            after_step_time_s=float(data.get("after_step_time_s", 0.0)),  # type: ignore[arg-type]
            before_memory_mb=float(data.get("before_memory_mb", 0.0)),  # type: ignore[arg-type]
            after_memory_mb=float(data.get("after_memory_mb", 0.0)),  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.8)),  # type: ignore[arg-type]
        )


@dataclass
class ROIEstimate:
    """ROI estimate for a single optimization scenario."""

    scenario_name: str
    time_saved_per_step_s: float
    speedup_factor: float
    time_saved_per_1000_steps_min: float
    time_saved_per_epoch_hours: float  # assuming steps_per_epoch
    cost_saved_per_hour: float
    cost_saved_per_day: float
    cost_saved_per_month: float
    memory_delta_mb: float
    confidence: float
    recommendation: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ROIEstimate:
        return cls(
            scenario_name=str(data.get("scenario_name", "")),
            time_saved_per_step_s=float(data.get("time_saved_per_step_s", 0.0)),  # type: ignore[arg-type]
            speedup_factor=float(data.get("speedup_factor", 0.0)),  # type: ignore[arg-type]
            time_saved_per_1000_steps_min=float(data.get("time_saved_per_1000_steps_min", 0.0)),  # type: ignore[arg-type]
            time_saved_per_epoch_hours=float(data.get("time_saved_per_epoch_hours", 0.0)),  # type: ignore[arg-type]
            cost_saved_per_hour=float(data.get("cost_saved_per_hour", 0.0)),  # type: ignore[arg-type]
            cost_saved_per_day=float(data.get("cost_saved_per_day", 0.0)),  # type: ignore[arg-type]
            cost_saved_per_month=float(data.get("cost_saved_per_month", 0.0)),  # type: ignore[arg-type]
            memory_delta_mb=float(data.get("memory_delta_mb", 0.0)),  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.0)),  # type: ignore[arg-type]
            recommendation=str(data.get("recommendation", "")),
        )


@dataclass
class ROIReport:
    """Full ROI analysis across all optimization scenarios."""

    estimates: List[ROIEstimate]
    total_speedup_factor: float
    total_cost_saved_per_month: float
    total_time_saved_per_epoch_hours: float
    gpu_type: str
    gpu_count: int
    steps_per_epoch: int
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "estimates": [e.to_dict() for e in self.estimates],
            "total_speedup_factor": self.total_speedup_factor,
            "total_cost_saved_per_month": self.total_cost_saved_per_month,
            "total_time_saved_per_epoch_hours": self.total_time_saved_per_epoch_hours,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "steps_per_epoch": self.steps_per_epoch,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ROIReport:
        raw_est = data.get("estimates", [])
        estimates = [ROIEstimate.from_dict(e) for e in raw_est]  # type: ignore[union-attr]
        return cls(
            estimates=estimates,
            total_speedup_factor=float(data.get("total_speedup_factor", 0.0)),  # type: ignore[arg-type]
            total_cost_saved_per_month=float(data.get("total_cost_saved_per_month", 0.0)),  # type: ignore[arg-type]
            total_time_saved_per_epoch_hours=float(data.get("total_time_saved_per_epoch_hours", 0.0)),  # type: ignore[arg-type]
            gpu_type=str(data.get("gpu_type", "")),
            gpu_count=int(data.get("gpu_count", 1)),  # type: ignore[arg-type]
            steps_per_epoch=int(data.get("steps_per_epoch", 0)),  # type: ignore[arg-type]
            summary=str(data.get("summary", "")),
        )


# ============================================================================
# ROI Calculator
# ============================================================================


class ROICalculator:
    """Calculate ROI of implementing profiler recommendations.

    Parameters
    ----------
    gpu_type:
        GPU type key from ``GPU_HOURLY_COST`` (e.g. "h100_80gb").
    gpu_count:
        Number of GPUs in the training job.
    steps_per_epoch:
        Number of training steps per epoch.
    hours_per_day:
        Hours of GPU usage per day (for cost projections).
    """

    def __init__(
        self,
        gpu_type: str = "a100_80gb",
        gpu_count: int = 1,
        steps_per_epoch: int = 10_000,
        hours_per_day: float = 24.0,
    ) -> None:
        self._gpu_type = gpu_type
        self._gpu_count = gpu_count
        self._steps_per_epoch = steps_per_epoch
        self._hours_per_day = hours_per_day
        self._hourly_cost = GPU_HOURLY_COST.get(gpu_type, 4.10) * gpu_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, scenario: OptimizationScenario) -> ROIEstimate:
        """Compute ROI for a single optimization scenario."""
        before = scenario.before_step_time_s
        after = scenario.after_step_time_s

        if before <= 0 or after <= 0:
            return self._empty_estimate(scenario.name, scenario.confidence)

        time_saved_per_step = before - after
        speedup = before / after

        # Time projections
        time_saved_1000_steps_min = (time_saved_per_step * 1000) / 60.0
        epoch_time_before_h = (before * self._steps_per_epoch) / 3600.0
        epoch_time_after_h = (after * self._steps_per_epoch) / 3600.0
        time_saved_epoch_h = epoch_time_before_h - epoch_time_after_h

        # Cost projections
        frac_saved = time_saved_per_step / before if before > 0 else 0.0
        cost_saved_per_hour = self._hourly_cost * frac_saved
        cost_saved_per_day = cost_saved_per_hour * self._hours_per_day
        cost_saved_per_month = cost_saved_per_day * 30

        mem_delta = scenario.after_memory_mb - scenario.before_memory_mb

        # Recommendation text
        if speedup >= 1.5:
            rec = f"Strongly recommended: {speedup:.1f}x speedup, saves ${cost_saved_per_month:.0f}/month"
        elif speedup >= 1.1:
            rec = f"Recommended: {speedup:.2f}x speedup, saves ${cost_saved_per_month:.0f}/month"
        elif speedup > 1.0:
            rec = f"Minor improvement: {speedup:.2f}x speedup, saves ${cost_saved_per_month:.0f}/month"
        else:
            rec = f"Not recommended: {speedup:.2f}x (regression)"

        return ROIEstimate(
            scenario_name=scenario.name,
            time_saved_per_step_s=time_saved_per_step,
            speedup_factor=speedup,
            time_saved_per_1000_steps_min=time_saved_1000_steps_min,
            time_saved_per_epoch_hours=time_saved_epoch_h,
            cost_saved_per_hour=cost_saved_per_hour,
            cost_saved_per_day=cost_saved_per_day,
            cost_saved_per_month=cost_saved_per_month,
            memory_delta_mb=mem_delta,
            confidence=scenario.confidence,
            recommendation=rec,
        )

    def analyze(self, scenarios: List[OptimizationScenario]) -> ROIReport:
        """Analyze all scenarios and produce a combined ROI report."""
        if not scenarios:
            return ROIReport(
                estimates=[],
                total_speedup_factor=1.0,
                total_cost_saved_per_month=0.0,
                total_time_saved_per_epoch_hours=0.0,
                gpu_type=self._gpu_type,
                gpu_count=self._gpu_count,
                steps_per_epoch=self._steps_per_epoch,
                summary="No optimization scenarios provided.",
            )

        estimates = [self.estimate(s) for s in scenarios]

        # Compound speedup (multiply individual speedups)
        total_speedup = 1.0
        for e in estimates:
            if e.speedup_factor > 0:
                total_speedup *= e.speedup_factor

        total_cost = sum(e.cost_saved_per_month for e in estimates)
        total_time = sum(e.time_saved_per_epoch_hours for e in estimates)

        # Build summary
        beneficial = [e for e in estimates if e.speedup_factor > 1.0]
        summary_parts = [
            f"{len(beneficial)}/{len(estimates)} optimizations recommended.",
            f"Combined speedup: {total_speedup:.2f}x.",
            f"Estimated savings: ${total_cost:.0f}/month on {self._gpu_count}x {self._gpu_type}.",
        ]
        if total_time > 0:
            summary_parts.append(
                f"Time saved per epoch: {total_time:.1f} hours."
            )

        return ROIReport(
            estimates=estimates,
            total_speedup_factor=total_speedup,
            total_cost_saved_per_month=total_cost,
            total_time_saved_per_epoch_hours=total_time,
            gpu_type=self._gpu_type,
            gpu_count=self._gpu_count,
            steps_per_epoch=self._steps_per_epoch,
            summary=" ".join(summary_parts),
        )

    @staticmethod
    def batch_size_roi(
        current_batch_size: int,
        proposed_batch_size: int,
        current_step_time_s: float,
        gpu_type: str = "a100_80gb",
        gpu_count: int = 1,
        steps_per_epoch: int = 10_000,
    ) -> ROIEstimate:
        """Quick helper: ROI from a batch size change.

        Larger batches process more samples per step. Step time grows
        sub-linearly (GPU parallelism absorbs part of the increase),
        so the *effective* time per sample drops. We model this with
        an 85% scaling efficiency factor and convert to an equivalent
        per-step speedup that accounts for fewer steps being needed.
        """
        if current_batch_size <= 0 or proposed_batch_size <= 0:
            calc = ROICalculator(gpu_type, gpu_count, steps_per_epoch)
            return calc._empty_estimate("batch_size_change", 0.5)

        ratio = proposed_batch_size / current_batch_size
        # Step time scales sub-linearly: ~15% overhead per 2x increase
        scaling_efficiency = 0.85
        new_step_time = current_step_time_s * (1.0 + (ratio - 1.0) * (1.0 - scaling_efficiency))

        # Total epoch time: steps * step_time.  Steps shrink by 1/ratio.
        before_epoch_time = current_step_time_s * steps_per_epoch
        adjusted_steps = steps_per_epoch / ratio
        after_epoch_time = new_step_time * adjusted_steps

        # Express the net saving as an equivalent step-time reduction
        # over the original number of steps so the standard estimate()
        # maths produce the right cost / time projections.
        equivalent_after_step_time = after_epoch_time / steps_per_epoch

        scenario = OptimizationScenario(
            name=f"batch_size_{current_batch_size}_to_{proposed_batch_size}",
            category="batch_size",
            before_value=f"batch_size={current_batch_size}",
            after_value=f"batch_size={proposed_batch_size}",
            before_step_time_s=current_step_time_s,
            after_step_time_s=equivalent_after_step_time,
            confidence=0.7,
        )

        calc = ROICalculator(gpu_type, gpu_count, steps_per_epoch)
        return calc.estimate(scenario)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _empty_estimate(self, name: str, confidence: float) -> ROIEstimate:
        return ROIEstimate(
            scenario_name=name,
            time_saved_per_step_s=0.0,
            speedup_factor=1.0,
            time_saved_per_1000_steps_min=0.0,
            time_saved_per_epoch_hours=0.0,
            cost_saved_per_hour=0.0,
            cost_saved_per_day=0.0,
            cost_saved_per_month=0.0,
            memory_delta_mb=0.0,
            confidence=confidence,
            recommendation="Insufficient data to estimate ROI",
        )
