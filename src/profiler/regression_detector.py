"""Performance regression detector.

Compares the current profiling run to historical baselines and detects
regressions in throughput, memory usage, GPU utilization, and
communication overhead.

Baselines are stored as simple JSON files so they can be committed to
version control or stored in artifact registries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Default percentage thresholds for flagging regressions.
_THROUGHPUT_REGRESSION_PCT: float = 5.0
_MEMORY_REGRESSION_PCT: float = 10.0
_UTILIZATION_REGRESSION_PCT: float = 10.0
_STEP_TIME_REGRESSION_PCT: float = 5.0


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class Baseline:
    """Historical baseline captured from a previous profiling run."""

    name: str
    timestamp: str  # ISO-8601
    avg_step_time_s: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    avg_gpu_utilization_pct: float
    avg_communication_overhead_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> Baseline:
        return cls(
            name=str(data.get("name", "")),
            timestamp=str(data.get("timestamp", "")),
            avg_step_time_s=float(data.get("avg_step_time_s", 0.0)),  # type: ignore[arg-type]
            throughput_samples_per_sec=float(data.get("throughput_samples_per_sec", 0.0)),  # type: ignore[arg-type]
            peak_memory_mb=float(data.get("peak_memory_mb", 0.0)),  # type: ignore[arg-type]
            avg_gpu_utilization_pct=float(data.get("avg_gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            avg_communication_overhead_pct=float(data.get("avg_communication_overhead_pct", 0.0)),  # type: ignore[arg-type]
            metadata=dict(data.get("metadata", {})),  # type: ignore[arg-type]
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Baseline:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True)
class RegressionFlag:
    """A single detected regression or improvement."""

    metric: str
    baseline_value: float
    current_value: float
    delta_pct: float
    severity: str  # "regression", "improvement", "unchanged"
    unit: str
    lower_is_better: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RegressionFlag:
        return cls(
            metric=str(data.get("metric", "")),
            baseline_value=float(data.get("baseline_value", 0.0)),  # type: ignore[arg-type]
            current_value=float(data.get("current_value", 0.0)),  # type: ignore[arg-type]
            delta_pct=float(data.get("delta_pct", 0.0)),  # type: ignore[arg-type]
            severity=str(data.get("severity", "unchanged")),
            unit=str(data.get("unit", "")),
            lower_is_better=bool(data.get("lower_is_better", True)),
        )


@dataclass
class RegressionReport:
    """Full regression detection report."""

    baseline_name: str
    flags: List[RegressionFlag]
    has_regressions: bool = False
    regression_count: int = 0
    improvement_count: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "has_regressions": self.has_regressions,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "summary": self.summary,
            "flags": [f.to_dict() for f in self.flags],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RegressionReport:
        raw_flags = data.get("flags", [])
        flags = [RegressionFlag.from_dict(f) for f in raw_flags]  # type: ignore[union-attr]
        return cls(
            baseline_name=str(data.get("baseline_name", "")),
            flags=flags,
        )


# ============================================================================
# Current run snapshot
# ============================================================================


@dataclass(frozen=True)
class CurrentRunMetrics:
    """Metrics from the current profiling run to compare against baseline."""

    avg_step_time_s: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    avg_gpu_utilization_pct: float
    avg_communication_overhead_pct: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CurrentRunMetrics:
        return cls(
            avg_step_time_s=float(data.get("avg_step_time_s", 0.0)),  # type: ignore[arg-type]
            throughput_samples_per_sec=float(data.get("throughput_samples_per_sec", 0.0)),  # type: ignore[arg-type]
            peak_memory_mb=float(data.get("peak_memory_mb", 0.0)),  # type: ignore[arg-type]
            avg_gpu_utilization_pct=float(data.get("avg_gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            avg_communication_overhead_pct=float(data.get("avg_communication_overhead_pct", 0.0)),  # type: ignore[arg-type]
        )

    @classmethod
    def from_gpu_profile(
        cls,
        result: GPUProfileResult,
        throughput: float = 0.0,
        step_time: float = 0.0,
        comm_overhead: float = 0.0,
    ) -> CurrentRunMetrics:
        return cls(
            avg_step_time_s=step_time if step_time > 0 else result.duration_s,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=result.peak_memory_mb,
            avg_gpu_utilization_pct=result.avg_utilization,
            avg_communication_overhead_pct=comm_overhead,
        )


# ============================================================================
# Regression Detector
# ============================================================================


class RegressionDetector:
    """Compare current run metrics against a historical baseline.

    Parameters
    ----------
    throughput_threshold_pct:
        Percentage drop in throughput to flag as regression.
    memory_threshold_pct:
        Percentage increase in memory to flag as regression.
    utilization_threshold_pct:
        Percentage drop in GPU utilization to flag as regression.
    step_time_threshold_pct:
        Percentage increase in step time to flag as regression.
    """

    def __init__(
        self,
        throughput_threshold_pct: float = _THROUGHPUT_REGRESSION_PCT,
        memory_threshold_pct: float = _MEMORY_REGRESSION_PCT,
        utilization_threshold_pct: float = _UTILIZATION_REGRESSION_PCT,
        step_time_threshold_pct: float = _STEP_TIME_REGRESSION_PCT,
    ) -> None:
        self._throughput_pct = throughput_threshold_pct
        self._memory_pct = memory_threshold_pct
        self._utilization_pct = utilization_threshold_pct
        self._step_time_pct = step_time_threshold_pct

    def compare(
        self,
        baseline: Baseline,
        current: CurrentRunMetrics,
    ) -> RegressionReport:
        """Run a full comparison and return a regression report."""
        flags: List[RegressionFlag] = []

        # Throughput (higher is better)
        flags.append(
            self._check_metric(
                metric="throughput",
                baseline_val=baseline.throughput_samples_per_sec,
                current_val=current.throughput_samples_per_sec,
                threshold_pct=self._throughput_pct,
                unit="samples/sec",
                lower_is_better=False,
            )
        )

        # Step time (lower is better)
        flags.append(
            self._check_metric(
                metric="avg_step_time",
                baseline_val=baseline.avg_step_time_s,
                current_val=current.avg_step_time_s,
                threshold_pct=self._step_time_pct,
                unit="seconds",
                lower_is_better=True,
            )
        )

        # Peak memory (lower is better)
        flags.append(
            self._check_metric(
                metric="peak_memory",
                baseline_val=baseline.peak_memory_mb,
                current_val=current.peak_memory_mb,
                threshold_pct=self._memory_pct,
                unit="MB",
                lower_is_better=True,
            )
        )

        # GPU utilization (higher is better)
        flags.append(
            self._check_metric(
                metric="avg_gpu_utilization",
                baseline_val=baseline.avg_gpu_utilization_pct,
                current_val=current.avg_gpu_utilization_pct,
                threshold_pct=self._utilization_pct,
                unit="%",
                lower_is_better=False,
            )
        )

        # Communication overhead (lower is better)
        if baseline.avg_communication_overhead_pct > 0 or current.avg_communication_overhead_pct > 0:
            flags.append(
                self._check_metric(
                    metric="communication_overhead",
                    baseline_val=baseline.avg_communication_overhead_pct,
                    current_val=current.avg_communication_overhead_pct,
                    threshold_pct=self._step_time_pct,
                    unit="%",
                    lower_is_better=True,
                )
            )

        regressions = [f for f in flags if f.severity == "regression"]
        improvements = [f for f in flags if f.severity == "improvement"]

        summary_parts: List[str] = []
        if regressions:
            names = ", ".join(f.metric for f in regressions)
            summary_parts.append(f"REGRESSIONS detected in: {names}")
        if improvements:
            names = ", ".join(f.metric for f in improvements)
            summary_parts.append(f"Improvements in: {names}")
        if not regressions and not improvements:
            summary_parts.append("No significant changes from baseline.")

        return RegressionReport(
            baseline_name=baseline.name,
            flags=flags,
            has_regressions=len(regressions) > 0,
            regression_count=len(regressions),
            improvement_count=len(improvements),
            summary="; ".join(summary_parts),
        )

    def compare_from_file(
        self,
        baseline_path: Path,
        current: CurrentRunMetrics,
    ) -> RegressionReport:
        """Load baseline from file and compare."""
        baseline = Baseline.load(baseline_path)
        return self.compare(baseline, current)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_metric(
        metric: str,
        baseline_val: float,
        current_val: float,
        threshold_pct: float,
        unit: str,
        lower_is_better: bool,
    ) -> RegressionFlag:
        if baseline_val == 0:
            return RegressionFlag(
                metric=metric,
                baseline_value=baseline_val,
                current_value=current_val,
                delta_pct=0.0,
                severity="unchanged",
                unit=unit,
                lower_is_better=lower_is_better,
            )

        delta_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100

        if lower_is_better:
            # Increase is bad
            if delta_pct > threshold_pct:
                severity = "regression"
            elif delta_pct < -threshold_pct:
                severity = "improvement"
            else:
                severity = "unchanged"
        else:
            # Decrease is bad
            if delta_pct < -threshold_pct:
                severity = "regression"
            elif delta_pct > threshold_pct:
                severity = "improvement"
            else:
                severity = "unchanged"

        return RegressionFlag(
            metric=metric,
            baseline_value=baseline_val,
            current_value=current_val,
            delta_pct=delta_pct,
            severity=severity,
            unit=unit,
            lower_is_better=lower_is_better,
        )
