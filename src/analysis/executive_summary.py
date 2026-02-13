"""Executive summary generator for profiling results.

Produces a concise, one-page summary aimed at non-technical stakeholders
(engineering managers, VPs, CTOs).  Focuses on business impact: cost,
time savings opportunity, and top actionable recommendations.

Consumes outputs from other profiler/analysis modules and distills them
into plain-language findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult
from src.profiler.memory_analyzer import MemoryAnalysisResult
from src.analysis.cost_analyzer import CostAnalysisResult

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class Recommendation:
    """A single actionable recommendation with estimated impact."""

    title: str
    description: str
    estimated_savings_pct: float
    effort: str  # "low", "medium", "high"
    priority: int  # 1 = highest

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> Recommendation:
        return cls(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            estimated_savings_pct=float(data.get("estimated_savings_pct", 0.0)),  # type: ignore[arg-type]
            effort=str(data.get("effort", "medium")),
            priority=int(data.get("priority", 99)),  # type: ignore[arg-type]
        )


@dataclass
class ExecutiveSummary:
    """One-page executive summary of profiling results."""

    # Headline numbers.
    gpu_utilization_pct: float
    monthly_gpu_cost: float
    potential_monthly_savings: float
    potential_savings_pct: float

    # Status assessments.
    memory_health: str  # "healthy", "warning", "critical"
    io_health: str
    overall_health: str

    # Top recommendations (max 3).
    top_recommendations: List[Recommendation]

    # Plain-text summary paragraphs.
    summary_text: str
    key_findings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "monthly_gpu_cost": self.monthly_gpu_cost,
            "potential_monthly_savings": self.potential_monthly_savings,
            "potential_savings_pct": self.potential_savings_pct,
            "memory_health": self.memory_health,
            "io_health": self.io_health,
            "overall_health": self.overall_health,
            "top_recommendations": [r.to_dict() for r in self.top_recommendations],
            "summary_text": self.summary_text,
            "key_findings": list(self.key_findings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ExecutiveSummary:
        raw_recs = data.get("top_recommendations", [])
        recs = [
            Recommendation.from_dict(r)  # type: ignore[arg-type]
            for r in raw_recs  # type: ignore[union-attr]
        ] if isinstance(raw_recs, list) else []

        raw_findings = data.get("key_findings", [])
        findings = [str(f) for f in raw_findings] if isinstance(raw_findings, list) else []  # type: ignore[union-attr]

        return cls(
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            monthly_gpu_cost=float(data.get("monthly_gpu_cost", 0.0)),  # type: ignore[arg-type]
            potential_monthly_savings=float(data.get("potential_monthly_savings", 0.0)),  # type: ignore[arg-type]
            potential_savings_pct=float(data.get("potential_savings_pct", 0.0)),  # type: ignore[arg-type]
            memory_health=str(data.get("memory_health", "unknown")),
            io_health=str(data.get("io_health", "unknown")),
            overall_health=str(data.get("overall_health", "unknown")),
            top_recommendations=recs,
            summary_text=str(data.get("summary_text", "")),
            key_findings=findings,
        )


# ============================================================================
# Executive Summary Generator
# ============================================================================


class ExecutiveSummaryGenerator:
    """Generate a one-page executive summary from profiling data.

    Usage::

        gen = ExecutiveSummaryGenerator(
            gpu_result=gpu_result,
            cost_result=cost_result,
            memory_result=memory_result,
        )
        summary = gen.generate()
        print(summary.summary_text)
    """

    def __init__(
        self,
        gpu_result: Optional[GPUProfileResult] = None,
        cost_result: Optional[CostAnalysisResult] = None,
        memory_result: Optional[MemoryAnalysisResult] = None,
        num_gpus: int = 1,
        gpu_name: str = "GPU",
    ) -> None:
        self._gpu_result = gpu_result
        self._cost_result = cost_result
        self._memory_result = memory_result
        self._num_gpus = max(num_gpus, 1)
        self._gpu_name = gpu_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> ExecutiveSummary:
        """Generate the executive summary."""
        utilization = self._get_utilization()
        monthly_cost = self._get_monthly_cost()
        savings = self._get_potential_savings()
        savings_pct = (savings / monthly_cost * 100.0) if monthly_cost > 0 else 0.0

        memory_health = self._assess_memory_health()
        io_health = "healthy"  # Default; updated if IO profiler data available
        overall_health = self._assess_overall_health(utilization, memory_health)

        recommendations = self._build_recommendations(utilization, memory_health)
        findings = self._build_findings(utilization, monthly_cost, savings, memory_health)
        summary_text = self._build_summary_text(
            utilization, monthly_cost, savings, savings_pct,
            memory_health, recommendations,
        )

        return ExecutiveSummary(
            gpu_utilization_pct=round(utilization, 1),
            monthly_gpu_cost=round(monthly_cost, 2),
            potential_monthly_savings=round(savings, 2),
            potential_savings_pct=round(savings_pct, 1),
            memory_health=memory_health,
            io_health=io_health,
            overall_health=overall_health,
            top_recommendations=recommendations[:3],
            summary_text=summary_text,
            key_findings=findings,
        )

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def _get_utilization(self) -> float:
        if self._gpu_result is not None:
            return self._gpu_result.avg_utilization
        return 0.0

    def _get_monthly_cost(self) -> float:
        if self._cost_result is not None:
            hourly = self._cost_result.current_estimate.total_cost_per_hour
            return hourly * 730.0  # ~hours/month
        return 0.0

    def _get_potential_savings(self) -> float:
        if self._cost_result is not None:
            return self._cost_result.potential_monthly_savings
        return 0.0

    # ------------------------------------------------------------------
    # Health assessments
    # ------------------------------------------------------------------

    def _assess_memory_health(self) -> str:
        if self._memory_result is None:
            return "unknown"

        if self._memory_result.leak_detected:
            return "critical"

        oom_risk = self._memory_result.oom_risk.risk_score
        if oom_risk > 0.8:
            return "critical"
        elif oom_risk > 0.5:
            return "warning"
        return "healthy"

    @staticmethod
    def _assess_overall_health(utilization: float, memory_health: str) -> str:
        if memory_health == "critical":
            return "critical"
        if utilization < 30.0:
            return "warning"
        if memory_health == "warning":
            return "warning"
        if utilization < 50.0:
            return "warning"
        return "healthy"

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self, utilization: float, memory_health: str,
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []
        priority = 1

        # Low utilization.
        if utilization < 50.0 and utilization > 0:
            waste_pct = 100.0 - utilization
            recs.append(Recommendation(
                title="Increase GPU utilization",
                description=(
                    f"GPUs are only {utilization:.0f}% utilized on average, "
                    f"wasting ~{waste_pct:.0f}% of compute capacity. "
                    f"Consider increasing batch size, reducing data loading "
                    f"bottlenecks, or consolidating workloads."
                ),
                estimated_savings_pct=round(waste_pct * 0.3, 1),
                effort="medium",
                priority=priority,
            ))
            priority += 1

        # Cost savings from GPU switching.
        if self._cost_result and self._cost_result.recommendations:
            best = self._cost_result.recommendations[0]
            recs.append(Recommendation(
                title=f"Switch to {best.gpu_name}",
                description=(
                    f"Switching from current GPU to {best.gpu_name} "
                    f"({best.num_gpus} GPU(s)) could save ~{best.estimated_savings_pct:.0f}% "
                    f"on compute costs."
                ),
                estimated_savings_pct=best.estimated_savings_pct,
                effort="high",
                priority=priority,
            ))
            priority += 1

        # Memory health.
        if memory_health == "critical":
            recs.append(Recommendation(
                title="Address memory issues",
                description=(
                    "Memory leak detected or OOM risk is critical. "
                    "This can cause training crashes and lost GPU-hours. "
                    "Investigate memory allocation patterns immediately."
                ),
                estimated_savings_pct=10.0,
                effort="medium",
                priority=1,  # Always highest priority
            ))
        elif memory_health == "warning":
            recs.append(Recommendation(
                title="Monitor memory usage",
                description=(
                    "Memory usage is elevated. Consider enabling gradient "
                    "checkpointing or mixed precision training to reduce "
                    "memory pressure."
                ),
                estimated_savings_pct=5.0,
                effort="low",
                priority=priority,
            ))
            priority += 1

        # Sort by priority.
        recs.sort(key=lambda r: r.priority)
        return recs

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def _build_findings(
        self,
        utilization: float,
        monthly_cost: float,
        savings: float,
        memory_health: str,
    ) -> List[str]:
        findings: List[str] = []

        if utilization > 0:
            findings.append(
                f"Average GPU utilization: {utilization:.0f}% "
                f"across {self._num_gpus} {self._gpu_name}(s)."
            )

        if monthly_cost > 0:
            findings.append(f"Estimated monthly GPU cost: ${monthly_cost:,.0f}.")

        if savings > 0:
            findings.append(
                f"Potential monthly savings: ${savings:,.0f} "
                f"through infrastructure optimization."
            )

        if memory_health == "critical":
            findings.append(
                "Memory health is CRITICAL — memory leaks or high OOM risk detected."
            )
        elif memory_health == "warning":
            findings.append(
                "Memory usage is elevated — monitor for potential OOM issues."
            )

        if utilization > 0 and utilization < 50:
            findings.append(
                f"GPU utilization is below 50%. Significant compute resources "
                f"are being wasted."
            )

        return findings

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def _build_summary_text(
        self,
        utilization: float,
        monthly_cost: float,
        savings: float,
        savings_pct: float,
        memory_health: str,
        recommendations: List[Recommendation],
    ) -> str:
        parts: List[str] = []

        parts.append("PROFILING EXECUTIVE SUMMARY")
        parts.append("=" * 40)
        parts.append("")

        # Headline.
        if monthly_cost > 0:
            parts.append(
                f"Your {self._num_gpus}x {self._gpu_name} cluster costs "
                f"~${monthly_cost:,.0f}/month."
            )
        if savings > 0:
            parts.append(
                f"We identified ${savings:,.0f}/month ({savings_pct:.0f}%) "
                f"in potential savings."
            )
        parts.append("")

        # Key metrics.
        parts.append("KEY METRICS:")
        if utilization > 0:
            parts.append(f"  GPU Utilization: {utilization:.0f}%")
        parts.append(f"  Memory Health: {memory_health.upper()}")
        if monthly_cost > 0:
            parts.append(f"  Monthly Cost: ${monthly_cost:,.0f}")
        if savings > 0:
            parts.append(f"  Savings Opportunity: ${savings:,.0f}/month")
        parts.append("")

        # Top recommendations.
        if recommendations:
            parts.append("TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                parts.append(f"  {i}. {rec.title}")
                parts.append(f"     {rec.description}")
                parts.append(f"     Estimated impact: ~{rec.estimated_savings_pct:.0f}% | Effort: {rec.effort}")
                parts.append("")

        return "\n".join(parts)
