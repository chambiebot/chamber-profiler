"""GPU memory leak detector for long-running training workloads.

Monitors GPU memory allocation patterns over time to detect leaks —
situations where memory is allocated but never freed, causing gradual
growth that eventually leads to OOM.  Common causes include accumulating
computation graphs, caching tensors across iterations, and growing lists
of intermediate results.

Uses statistical trend analysis on memory snapshots rather than requiring
CUDA debugging tools, making it lightweight enough to run in production.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.profiler.memory_analyzer import MemorySnapshot

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class LeakCandidate:
    """A detected potential memory leak."""

    start_timestamp: float
    end_timestamp: float
    start_allocated_mb: float
    end_allocated_mb: float
    growth_mb: float
    growth_rate_mb_per_sec: float
    num_snapshots: int
    confidence: float  # 0.0 – 1.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> LeakCandidate:
        return cls(
            start_timestamp=float(data.get("start_timestamp", 0.0)),  # type: ignore[arg-type]
            end_timestamp=float(data.get("end_timestamp", 0.0)),  # type: ignore[arg-type]
            start_allocated_mb=float(data.get("start_allocated_mb", 0.0)),  # type: ignore[arg-type]
            end_allocated_mb=float(data.get("end_allocated_mb", 0.0)),  # type: ignore[arg-type]
            growth_mb=float(data.get("growth_mb", 0.0)),  # type: ignore[arg-type]
            growth_rate_mb_per_sec=float(data.get("growth_rate_mb_per_sec", 0.0)),  # type: ignore[arg-type]
            num_snapshots=int(data.get("num_snapshots", 0)),  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class LeakDetectionResult:
    """Full output of a memory leak detection analysis."""

    has_leak: bool
    leak_candidates: List[LeakCandidate]
    overall_growth_mb: float
    overall_growth_rate_mb_per_sec: float
    trend_slope: float  # MB per second (from linear regression)
    trend_r_squared: float  # goodness of fit
    recommendations: List[str]
    snapshots_analyzed: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "has_leak": self.has_leak,
            "leak_candidates": [lc.to_dict() for lc in self.leak_candidates],
            "overall_growth_mb": self.overall_growth_mb,
            "overall_growth_rate_mb_per_sec": self.overall_growth_rate_mb_per_sec,
            "trend_slope": self.trend_slope,
            "trend_r_squared": self.trend_r_squared,
            "recommendations": list(self.recommendations),
            "snapshots_analyzed": self.snapshots_analyzed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> LeakDetectionResult:
        raw_candidates = data.get("leak_candidates", [])
        candidates = [
            LeakCandidate.from_dict(c)  # type: ignore[arg-type]
            for c in raw_candidates  # type: ignore[union-attr]
        ] if isinstance(raw_candidates, list) else []

        raw_recs = data.get("recommendations", [])
        recs = [str(r) for r in raw_recs] if isinstance(raw_recs, list) else []  # type: ignore[union-attr]

        return cls(
            has_leak=bool(data.get("has_leak", False)),
            leak_candidates=candidates,
            overall_growth_mb=float(data.get("overall_growth_mb", 0.0)),  # type: ignore[arg-type]
            overall_growth_rate_mb_per_sec=float(data.get("overall_growth_rate_mb_per_sec", 0.0)),  # type: ignore[arg-type]
            trend_slope=float(data.get("trend_slope", 0.0)),  # type: ignore[arg-type]
            trend_r_squared=float(data.get("trend_r_squared", 0.0)),  # type: ignore[arg-type]
            recommendations=recs,
            snapshots_analyzed=int(data.get("snapshots_analyzed", 0)),  # type: ignore[arg-type]
        )


# ============================================================================
# Memory Leak Detector
# ============================================================================


class MemoryLeakDetector:
    """Detect GPU memory leaks from a series of memory snapshots.

    Uses two complementary approaches:
    1. **Monotonic run detection**: finds sustained runs of increasing
       allocation (like MemoryAnalyzer but with richer output).
    2. **Linear regression trend analysis**: fits a line to the allocation
       time series and flags positive slopes with high R² as leaks.

    Usage::

        detector = MemoryLeakDetector()
        result = detector.analyze(snapshots)
        if result.has_leak:
            for candidate in result.leak_candidates:
                print(f"Leak: {candidate.growth_mb:.1f} MB over "
                      f"{candidate.num_snapshots} samples")
    """

    # Minimum number of snapshots required for analysis.
    MIN_SNAPSHOTS: int = 5

    # Minimum consecutive increases to flag a monotonic run.
    MONOTONIC_RUN_LENGTH: int = 5

    # Minimum growth in MB over a monotonic run to count.
    MIN_RUN_GROWTH_MB: float = 1.0

    # R² threshold for trend-based leak detection.
    R_SQUARED_THRESHOLD: float = 0.7

    # Minimum positive slope (MB/s) to consider a trend a leak.
    MIN_SLOPE_MB_PER_SEC: float = 0.001

    def __init__(
        self,
        min_run_length: int = 5,
        min_growth_mb: float = 1.0,
        r_squared_threshold: float = 0.7,
    ) -> None:
        self._min_run_length = max(min_run_length, 3)
        self._min_growth_mb = min_growth_mb
        self._r_squared_threshold = r_squared_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, snapshots: List[MemorySnapshot]) -> LeakDetectionResult:
        """Run leak detection on the given snapshot series.

        Parameters
        ----------
        snapshots:
            Ordered list of memory snapshots (by timestamp).

        Returns
        -------
        LeakDetectionResult
        """
        if len(snapshots) < self.MIN_SNAPSHOTS:
            return self._empty_result(len(snapshots))

        # Extract time series.
        times = [s.timestamp for s in snapshots]
        allocs = [s.allocated_mb for s in snapshots]

        # 1. Monotonic run detection.
        candidates = self._detect_monotonic_runs(snapshots)

        # 2. Linear regression trend.
        slope, r_squared = self._linear_regression(times, allocs)

        # Overall growth.
        overall_growth = allocs[-1] - allocs[0]
        duration = times[-1] - times[0]
        growth_rate = overall_growth / duration if duration > 0 else 0.0

        # Determine if there's a leak.
        has_leak = False
        if candidates:
            has_leak = True
        if slope > self.MIN_SLOPE_MB_PER_SEC and r_squared >= self._r_squared_threshold:
            has_leak = True

        recommendations = self._generate_recommendations(
            has_leak, candidates, slope, r_squared, overall_growth, duration,
        )

        return LeakDetectionResult(
            has_leak=has_leak,
            leak_candidates=candidates,
            overall_growth_mb=round(overall_growth, 3),
            overall_growth_rate_mb_per_sec=round(growth_rate, 6),
            trend_slope=round(slope, 6),
            trend_r_squared=round(r_squared, 4),
            recommendations=recommendations,
            snapshots_analyzed=len(snapshots),
        )

    # ------------------------------------------------------------------
    # Monotonic run detection
    # ------------------------------------------------------------------

    def _detect_monotonic_runs(
        self, snapshots: List[MemorySnapshot],
    ) -> List[LeakCandidate]:
        """Find sustained runs of strictly increasing allocation."""
        candidates: List[LeakCandidate] = []
        run_start = 0
        run_length = 1

        for i in range(1, len(snapshots)):
            if snapshots[i].allocated_mb > snapshots[i - 1].allocated_mb:
                run_length += 1
            else:
                if run_length >= self._min_run_length:
                    candidate = self._make_candidate(snapshots, run_start, i - 1)
                    if candidate is not None:
                        candidates.append(candidate)
                run_start = i
                run_length = 1

        # Check final run.
        if run_length >= self._min_run_length:
            candidate = self._make_candidate(snapshots, run_start, len(snapshots) - 1)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _make_candidate(
        self,
        snapshots: List[MemorySnapshot],
        start_idx: int,
        end_idx: int,
    ) -> Optional[LeakCandidate]:
        """Create a LeakCandidate from a monotonic run, if it meets thresholds."""
        growth = snapshots[end_idx].allocated_mb - snapshots[start_idx].allocated_mb
        if growth < self._min_growth_mb:
            return None

        duration = snapshots[end_idx].timestamp - snapshots[start_idx].timestamp
        rate = growth / duration if duration > 0 else 0.0
        num = end_idx - start_idx + 1

        # Confidence based on run length and growth magnitude.
        length_factor = min(num / 20.0, 1.0)
        growth_factor = min(growth / 100.0, 1.0)
        confidence = 0.5 * length_factor + 0.5 * growth_factor

        return LeakCandidate(
            start_timestamp=snapshots[start_idx].timestamp,
            end_timestamp=snapshots[end_idx].timestamp,
            start_allocated_mb=snapshots[start_idx].allocated_mb,
            end_allocated_mb=snapshots[end_idx].allocated_mb,
            growth_mb=round(growth, 3),
            growth_rate_mb_per_sec=round(rate, 6),
            num_snapshots=num,
            confidence=round(min(confidence, 1.0), 4),
        )

    # ------------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_regression(
        x: List[float], y: List[float],
    ) -> Tuple[float, float]:
        """Simple OLS linear regression. Returns (slope, r_squared).

        Pure Python — no numpy dependency.
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # R² calculation.
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot == 0:
            return slope, 1.0 if slope == 0 else 0.0

        intercept = (sum_y - slope * sum_x) / n
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
        r_squared = 1.0 - (ss_res / ss_tot)

        return slope, max(r_squared, 0.0)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(
        has_leak: bool,
        candidates: List[LeakCandidate],
        slope: float,
        r_squared: float,
        overall_growth: float,
        duration: float,
    ) -> List[str]:
        """Generate actionable recommendations based on leak analysis."""
        if not has_leak:
            return []

        recs: List[str] = []

        if overall_growth > 0 and duration > 0:
            hours_to_double = (overall_growth / (overall_growth / duration)) / 3600.0 if slope > 0 else float("inf")
            if hours_to_double < 24:
                recs.append(
                    f"Memory is growing at {slope * 1000:.2f} MB/s. "
                    f"At this rate, memory usage will double in ~{hours_to_double:.1f} hours."
                )

        recs.append(
            "Check for tensors being appended to lists without clearing "
            "(e.g., loss history, intermediate activations)."
        )
        recs.append(
            "Ensure .detach() is called on tensors stored for logging "
            "to avoid retaining the computation graph."
        )

        if len(candidates) > 1:
            recs.append(
                f"Detected {len(candidates)} separate leak episodes. "
                f"This may indicate multiple leak sources in different code paths."
            )

        if r_squared > 0.9:
            recs.append(
                "Memory growth is highly linear (R²={:.2f}), suggesting a "
                "per-iteration leak. Check the training loop body for "
                "accumulating state.".format(r_squared)
            )

        recs.append(
            "Use torch.cuda.memory_snapshot() to get detailed allocation "
            "traces for identifying the leaking code path."
        )

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(n_snapshots: int) -> LeakDetectionResult:
        return LeakDetectionResult(
            has_leak=False,
            leak_candidates=[],
            overall_growth_mb=0.0,
            overall_growth_rate_mb_per_sec=0.0,
            trend_slope=0.0,
            trend_r_squared=0.0,
            recommendations=[],
            snapshots_analyzed=n_snapshots,
        )
