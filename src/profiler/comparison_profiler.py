"""Comparison profiler for side-by-side analysis of two profiling runs.

Compares two profiling results and highlights regressions, improvements,
and changes across all key metrics. Produces a structured comparison
result that can be rendered as a terminal table or JSON report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTraceResult
from src.analysis.bottleneck_detector import BottleneckDetector, PerformanceSummary

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Minimum percentage change to flag as a regression or improvement.
_REGRESSION_THRESHOLD_PCT: float = 5.0
_IMPROVEMENT_THRESHOLD_PCT: float = 5.0


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class MetricDelta:
    """A single metric comparison between two profiles."""

    name: str
    value_a: float
    value_b: float
    delta: float
    delta_pct: float
    unit: str
    status: str  # "improved", "regressed", "unchanged"
    lower_is_better: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> MetricDelta:
        return cls(
            name=str(data.get("name", "")),
            value_a=float(data.get("value_a", 0.0)),  # type: ignore[arg-type]
            value_b=float(data.get("value_b", 0.0)),  # type: ignore[arg-type]
            delta=float(data.get("delta", 0.0)),  # type: ignore[arg-type]
            delta_pct=float(data.get("delta_pct", 0.0)),  # type: ignore[arg-type]
            unit=str(data.get("unit", "")),
            status=str(data.get("status", "unchanged")),
            lower_is_better=bool(data.get("lower_is_better", True)),
        )


@dataclass
class KernelDelta:
    """Comparison of a specific kernel between two runs."""

    kernel_name: str
    category: str
    duration_a_us: float
    duration_b_us: float
    delta_us: float
    delta_pct: float
    status: str  # "improved", "regressed", "new", "removed"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> KernelDelta:
        return cls(
            kernel_name=str(data.get("kernel_name", "")),
            category=str(data.get("category", "")),
            duration_a_us=float(data.get("duration_a_us", 0.0)),  # type: ignore[arg-type]
            duration_b_us=float(data.get("duration_b_us", 0.0)),  # type: ignore[arg-type]
            delta_us=float(data.get("delta_us", 0.0)),  # type: ignore[arg-type]
            delta_pct=float(data.get("delta_pct", 0.0)),  # type: ignore[arg-type]
            status=str(data.get("status", "unchanged")),
        )


@dataclass
class ComparisonResult:
    """Full comparison between two profiling runs."""

    label_a: str
    label_b: str
    metric_deltas: List[MetricDelta]
    kernel_deltas: List[KernelDelta]
    regressions: List[MetricDelta]
    improvements: List[MetricDelta]
    summary_a: Optional[PerformanceSummary]
    summary_b: Optional[PerformanceSummary]
    speedup: float  # > 1.0 means B is faster
    verdict: str  # human-readable verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "metric_deltas": [m.to_dict() for m in self.metric_deltas],
            "kernel_deltas": [k.to_dict() for k in self.kernel_deltas],
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "speedup": self.speedup,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComparisonResult:
        return cls(
            label_a=str(data.get("label_a", "A")),
            label_b=str(data.get("label_b", "B")),
            metric_deltas=[
                MetricDelta.from_dict(m)
                for m in data.get("metric_deltas", [])
            ],
            kernel_deltas=[
                KernelDelta.from_dict(k)
                for k in data.get("kernel_deltas", [])
            ],
            regressions=[
                MetricDelta.from_dict(r)
                for r in data.get("regressions", [])
            ],
            improvements=[
                MetricDelta.from_dict(i)
                for i in data.get("improvements", [])
            ],
            summary_a=None,
            summary_b=None,
            speedup=float(data.get("speedup", 1.0)),  # type: ignore[arg-type]
            verdict=str(data.get("verdict", "")),
        )


# ============================================================================
# Comparison Profiler
# ============================================================================


class ComparisonProfiler:
    """Compare two profiling runs and highlight regressions/improvements.

    Usage::

        profiler = ComparisonProfiler(
            gpu_result_a=baseline_gpu,
            gpu_result_b=optimized_gpu,
            kernel_result_a=baseline_kernels,
            kernel_result_b=optimized_kernels,
        )
        result = profiler.compare()
        for reg in result.regressions:
            print(f"REGRESSION: {reg.name} {reg.delta_pct:+.1f}%")
        for imp in result.improvements:
            print(f"IMPROVED: {imp.name} {imp.delta_pct:+.1f}%")
    """

    def __init__(
        self,
        gpu_result_a: Optional[GPUProfileResult] = None,
        gpu_result_b: Optional[GPUProfileResult] = None,
        kernel_result_a: Optional[KernelTraceResult] = None,
        kernel_result_b: Optional[KernelTraceResult] = None,
        summary_a: Optional[PerformanceSummary] = None,
        summary_b: Optional[PerformanceSummary] = None,
        label_a: str = "Baseline",
        label_b: str = "Current",
    ) -> None:
        self._gpu_a = gpu_result_a
        self._gpu_b = gpu_result_b
        self._kernel_a = kernel_result_a
        self._kernel_b = kernel_result_b
        self._summary_a = summary_a
        self._summary_b = summary_b
        self._label_a = label_a
        self._label_b = label_b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self) -> ComparisonResult:
        """Run the comparison and return a structured result."""
        # Build summaries if not provided.
        summary_a = self._summary_a
        summary_b = self._summary_b

        if summary_a is None:
            summary_a = self._build_summary(self._gpu_a, self._kernel_a)
        if summary_b is None:
            summary_b = self._build_summary(self._gpu_b, self._kernel_b)

        # Compare high-level metrics.
        metric_deltas = self._compare_metrics(summary_a, summary_b)

        # Compare kernels.
        kernel_deltas = self._compare_kernels()

        # Classify regressions and improvements.
        regressions = [
            m for m in metric_deltas if m.status == "regressed"
        ]
        improvements = [
            m for m in metric_deltas if m.status == "improved"
        ]

        # Compute speedup.
        speedup = 1.0
        if summary_a.total_time_s > 0 and summary_b.total_time_s > 0:
            speedup = summary_a.total_time_s / summary_b.total_time_s

        verdict = self._build_verdict(
            speedup, regressions, improvements, summary_a, summary_b,
        )

        return ComparisonResult(
            label_a=self._label_a,
            label_b=self._label_b,
            metric_deltas=metric_deltas,
            kernel_deltas=kernel_deltas,
            regressions=regressions,
            improvements=improvements,
            summary_a=summary_a,
            summary_b=summary_b,
            speedup=round(speedup, 3),
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Internal: metric comparison
    # ------------------------------------------------------------------

    def _compare_metrics(
        self,
        summary_a: PerformanceSummary,
        summary_b: PerformanceSummary,
    ) -> List[MetricDelta]:
        """Compare high-level metrics between two summaries."""
        deltas: List[MetricDelta] = []

        # Duration (lower is better).
        deltas.append(self._make_delta(
            "Total Duration",
            summary_a.total_time_s,
            summary_b.total_time_s,
            unit="s",
            lower_is_better=True,
        ))

        # GPU efficiency (higher is better).
        deltas.append(self._make_delta(
            "GPU Efficiency",
            summary_a.overall_efficiency * 100.0,
            summary_b.overall_efficiency * 100.0,
            unit="%",
            lower_is_better=False,
        ))

        # Compute % (higher is better).
        deltas.append(self._make_delta(
            "Compute %",
            summary_a.compute_pct,
            summary_b.compute_pct,
            unit="%",
            lower_is_better=False,
        ))

        # Idle % (lower is better).
        deltas.append(self._make_delta(
            "Idle %",
            summary_a.idle_pct,
            summary_b.idle_pct,
            unit="%",
            lower_is_better=True,
        ))

        # Communication % (lower is better).
        deltas.append(self._make_delta(
            "Communication %",
            summary_a.communication_pct,
            summary_b.communication_pct,
            unit="%",
            lower_is_better=True,
        ))

        # Memory % (lower is better).
        deltas.append(self._make_delta(
            "Memory %",
            summary_a.memory_pct,
            summary_b.memory_pct,
            unit="%",
            lower_is_better=True,
        ))

        # Data Loading % (lower is better).
        deltas.append(self._make_delta(
            "Data Loading %",
            summary_a.data_loading_pct,
            summary_b.data_loading_pct,
            unit="%",
            lower_is_better=True,
        ))

        # Bottleneck count (lower is better).
        deltas.append(self._make_delta(
            "Bottleneck Count",
            float(len(summary_a.bottlenecks)),
            float(len(summary_b.bottlenecks)),
            unit="",
            lower_is_better=True,
        ))

        # GPU utilization from raw profiles (higher is better).
        if self._gpu_a is not None and self._gpu_b is not None:
            deltas.append(self._make_delta(
                "Avg GPU Utilization",
                self._gpu_a.avg_utilization,
                self._gpu_b.avg_utilization,
                unit="%",
                lower_is_better=False,
            ))
            deltas.append(self._make_delta(
                "Peak Memory",
                self._gpu_a.peak_memory_mb,
                self._gpu_b.peak_memory_mb,
                unit="MB",
                lower_is_better=True,
            ))

        return deltas

    # ------------------------------------------------------------------
    # Internal: kernel comparison
    # ------------------------------------------------------------------

    def _compare_kernels(self) -> List[KernelDelta]:
        """Compare kernel-level data between two runs."""
        if self._kernel_a is None or self._kernel_b is None:
            return []

        # Aggregate kernel times by name in each run.
        times_a = self._aggregate_kernel_times(self._kernel_a)
        times_b = self._aggregate_kernel_times(self._kernel_b)

        all_names = set(times_a.keys()) | set(times_b.keys())
        deltas: List[KernelDelta] = []

        for name in all_names:
            dur_a, cat_a = times_a.get(name, (0.0, "other"))
            dur_b, cat_b = times_b.get(name, (0.0, "other"))
            category = cat_b if cat_b != "other" else cat_a

            delta_us = dur_b - dur_a
            if dur_a > 0:
                delta_pct = (delta_us / dur_a) * 100.0
            elif dur_b > 0:
                delta_pct = 100.0
            else:
                delta_pct = 0.0

            if dur_a == 0.0:
                status = "new"
            elif dur_b == 0.0:
                status = "removed"
            elif delta_pct < -_IMPROVEMENT_THRESHOLD_PCT:
                status = "improved"
            elif delta_pct > _REGRESSION_THRESHOLD_PCT:
                status = "regressed"
            else:
                status = "unchanged"

            deltas.append(KernelDelta(
                kernel_name=name,
                category=category,
                duration_a_us=round(dur_a, 2),
                duration_b_us=round(dur_b, 2),
                delta_us=round(delta_us, 2),
                delta_pct=round(delta_pct, 1),
                status=status,
            ))

        # Sort by absolute delta (largest regressions first).
        deltas.sort(key=lambda d: abs(d.delta_us), reverse=True)
        return deltas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_delta(
        name: str,
        value_a: float,
        value_b: float,
        unit: str,
        lower_is_better: bool,
    ) -> MetricDelta:
        """Create a MetricDelta with automatic status classification."""
        delta = value_b - value_a
        if abs(value_a) > 0.001:
            delta_pct = (delta / abs(value_a)) * 100.0
        else:
            delta_pct = 0.0 if abs(delta) < 0.001 else 100.0

        # Classify.
        if abs(delta_pct) < _REGRESSION_THRESHOLD_PCT:
            status = "unchanged"
        elif lower_is_better:
            status = "improved" if delta < 0 else "regressed"
        else:
            status = "improved" if delta > 0 else "regressed"

        return MetricDelta(
            name=name,
            value_a=round(value_a, 4),
            value_b=round(value_b, 4),
            delta=round(delta, 4),
            delta_pct=round(delta_pct, 1),
            unit=unit,
            status=status,
            lower_is_better=lower_is_better,
        )

    @staticmethod
    def _aggregate_kernel_times(
        result: KernelTraceResult,
    ) -> Dict[str, Tuple[float, str]]:
        """Aggregate kernel durations by name."""
        times: Dict[str, Tuple[float, str]] = {}
        for k in result.kernels:
            existing = times.get(k.name, (0.0, k.category))
            times[k.name] = (existing[0] + k.duration_us, k.category)
        return times

    @staticmethod
    def _build_summary(
        gpu_result: Optional[GPUProfileResult],
        kernel_result: Optional[KernelTraceResult],
    ) -> PerformanceSummary:
        """Build a PerformanceSummary from raw profiler results."""
        detector = BottleneckDetector(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
        )
        return detector.analyze()

    @staticmethod
    def _build_verdict(
        speedup: float,
        regressions: List[MetricDelta],
        improvements: List[MetricDelta],
        summary_a: PerformanceSummary,
        summary_b: PerformanceSummary,
    ) -> str:
        """Generate a human-readable verdict."""
        parts: List[str] = []

        if speedup > 1.05:
            parts.append(f"Performance improved: {speedup:.2f}x faster.")
        elif speedup < 0.95:
            parts.append(
                f"Performance regressed: {1.0 / speedup:.2f}x slower."
            )
        else:
            parts.append("Performance is roughly equivalent.")

        if regressions:
            reg_names = [r.name for r in regressions[:3]]
            parts.append(
                f"{len(regressions)} regression(s) detected: "
                f"{', '.join(reg_names)}."
            )

        if improvements:
            imp_names = [i.name for i in improvements[:3]]
            parts.append(
                f"{len(improvements)} improvement(s) detected: "
                f"{', '.join(imp_names)}."
            )

        if summary_a.primary_bottleneck != summary_b.primary_bottleneck:
            parts.append(
                f"Primary bottleneck changed from "
                f"'{summary_a.primary_bottleneck}' to "
                f"'{summary_b.primary_bottleneck}'."
            )

        return " ".join(parts)
