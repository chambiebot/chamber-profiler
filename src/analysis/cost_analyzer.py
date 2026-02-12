"""GPU cost analysis module.

Estimates the $/hour cost of a profiled GPU workload and recommends
cheaper or more cost-effective GPU configurations based on observed
utilization patterns.

Works purely from profiling data -- no cloud provider API access needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult

logger = logging.getLogger(__name__)


# ============================================================================
# GPU pricing database (on-demand $/hr per GPU, approximate)
# ============================================================================

# Cloud GPU pricing as of late 2024 / early 2025 (approximate on-demand).
GPU_CATALOG: Dict[str, Dict[str, Any]] = {
    "a100_40gb": {
        "name": "NVIDIA A100 40GB",
        "memory_gb": 40,
        "fp16_tflops": 312.0,
        "price_per_hour": 3.40,
        "provider": "AWS (p4d)",
    },
    "a100_80gb": {
        "name": "NVIDIA A100 80GB",
        "memory_gb": 80,
        "fp16_tflops": 312.0,
        "price_per_hour": 4.10,
        "provider": "AWS (p4de)",
    },
    "h100_80gb": {
        "name": "NVIDIA H100 80GB",
        "memory_gb": 80,
        "fp16_tflops": 989.5,
        "price_per_hour": 5.50,
        "provider": "AWS (p5)",
    },
    "a10g": {
        "name": "NVIDIA A10G",
        "memory_gb": 24,
        "fp16_tflops": 125.0,
        "price_per_hour": 1.20,
        "provider": "AWS (g5)",
    },
    "l4": {
        "name": "NVIDIA L4",
        "memory_gb": 24,
        "fp16_tflops": 121.0,
        "price_per_hour": 0.81,
        "provider": "GCP",
    },
    "l40s": {
        "name": "NVIDIA L40S",
        "memory_gb": 48,
        "fp16_tflops": 362.0,
        "price_per_hour": 2.70,
        "provider": "Various",
    },
    "t4": {
        "name": "NVIDIA T4",
        "memory_gb": 16,
        "fp16_tflops": 65.0,
        "price_per_hour": 0.53,
        "provider": "AWS (g4dn) / GCP",
    },
    "v100_16gb": {
        "name": "NVIDIA V100 16GB",
        "memory_gb": 16,
        "fp16_tflops": 125.0,
        "price_per_hour": 1.50,
        "provider": "AWS (p3)",
    },
    "rtx_4090": {
        "name": "NVIDIA RTX 4090",
        "memory_gb": 24,
        "fp16_tflops": 330.0,
        "price_per_hour": 0.74,
        "provider": "Cloud GPU providers",
    },
    "rtx_3090": {
        "name": "NVIDIA RTX 3090",
        "memory_gb": 24,
        "fp16_tflops": 142.0,
        "price_per_hour": 0.44,
        "provider": "Cloud GPU providers",
    },
}


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CostEstimate:
    """Cost estimate for the profiled workload."""

    gpu_name: str
    num_gpus: int
    cost_per_gpu_hour: float
    total_cost_per_hour: float
    profiled_duration_hours: float
    estimated_session_cost: float
    gpu_utilization_pct: float
    effective_cost_per_util_hour: float  # cost adjusted for utilization

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CostEstimate:
        return cls(
            gpu_name=str(data.get("gpu_name", "")),
            num_gpus=int(data.get("num_gpus", 0)),  # type: ignore[arg-type]
            cost_per_gpu_hour=float(data.get("cost_per_gpu_hour", 0.0)),  # type: ignore[arg-type]
            total_cost_per_hour=float(data.get("total_cost_per_hour", 0.0)),  # type: ignore[arg-type]
            profiled_duration_hours=float(data.get("profiled_duration_hours", 0.0)),  # type: ignore[arg-type]
            estimated_session_cost=float(data.get("estimated_session_cost", 0.0)),  # type: ignore[arg-type]
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            effective_cost_per_util_hour=float(data.get("effective_cost_per_util_hour", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class GPURecommendation:
    """A recommended alternative GPU configuration."""

    gpu_key: str
    gpu_name: str
    cost_per_gpu_hour: float
    estimated_cost_per_hour: float
    num_gpus: int
    memory_gb: int
    fp16_tflops: float
    estimated_savings_pct: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> GPURecommendation:
        return cls(
            gpu_key=str(data.get("gpu_key", "")),
            gpu_name=str(data.get("gpu_name", "")),
            cost_per_gpu_hour=float(data.get("cost_per_gpu_hour", 0.0)),  # type: ignore[arg-type]
            estimated_cost_per_hour=float(data.get("estimated_cost_per_hour", 0.0)),  # type: ignore[arg-type]
            num_gpus=int(data.get("num_gpus", 0)),  # type: ignore[arg-type]
            memory_gb=int(data.get("memory_gb", 0)),  # type: ignore[arg-type]
            fp16_tflops=float(data.get("fp16_tflops", 0.0)),  # type: ignore[arg-type]
            estimated_savings_pct=float(data.get("estimated_savings_pct", 0.0)),  # type: ignore[arg-type]
            reason=str(data.get("reason", "")),
        )


@dataclass
class CostAnalysisResult:
    """Full cost analysis output."""

    current_estimate: CostEstimate
    recommendations: List[GPURecommendation]
    potential_monthly_savings: float
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "current_estimate": self.current_estimate.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "potential_monthly_savings": self.potential_monthly_savings,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CostAnalysisResult:
        raw_est = data.get("current_estimate", {})
        current_estimate = CostEstimate.from_dict(raw_est) if isinstance(raw_est, dict) else CostEstimate(  # type: ignore[arg-type]
            gpu_name="", num_gpus=0, cost_per_gpu_hour=0.0,
            total_cost_per_hour=0.0, profiled_duration_hours=0.0,
            estimated_session_cost=0.0, gpu_utilization_pct=0.0,
            effective_cost_per_util_hour=0.0,
        )
        raw_recs = data.get("recommendations", [])
        recommendations = [
            GPURecommendation.from_dict(r)  # type: ignore[arg-type]
            for r in raw_recs  # type: ignore[union-attr]
        ] if isinstance(raw_recs, list) else []
        return cls(
            current_estimate=current_estimate,
            recommendations=recommendations,
            potential_monthly_savings=float(data.get("potential_monthly_savings", 0.0)),  # type: ignore[arg-type]
            summary=str(data.get("summary", "")),
        )


# ============================================================================
# Cost Analyzer
# ============================================================================


class CostAnalyzer:
    """Analyze GPU costs and recommend cheaper configurations.

    Usage::

        analyzer = CostAnalyzer(
            gpu_result=gpu_result,
            gpu_name="a100_80gb",
            num_gpus=4,
        )
        result = analyzer.analyze()
        print(result.current_estimate.total_cost_per_hour)
        for rec in result.recommendations:
            print(rec.gpu_name, rec.estimated_savings_pct)
    """

    def __init__(
        self,
        gpu_result: Optional[GPUProfileResult] = None,
        gpu_name: str = "a100_80gb",
        num_gpus: int = 1,
        cost_per_gpu_hour: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
    ) -> None:
        self._gpu_result = gpu_result
        self._gpu_name = gpu_name
        self._num_gpus = max(num_gpus, 1)
        self._custom_cost = cost_per_gpu_hour
        self._peak_memory_mb = peak_memory_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> CostAnalysisResult:
        """Run cost analysis and generate recommendations."""
        current = self._estimate_current_cost()
        recommendations = self._recommend_cheaper_configs(current)
        monthly_savings = self._estimate_monthly_savings(current, recommendations)
        summary = self._generate_summary(current, recommendations, monthly_savings)

        return CostAnalysisResult(
            current_estimate=current,
            recommendations=recommendations,
            potential_monthly_savings=round(monthly_savings, 2),
            summary=summary,
        )

    def estimate_cost(
        self,
        gpu_name: Optional[str] = None,
        num_gpus: Optional[int] = None,
        duration_hours: Optional[float] = None,
    ) -> CostEstimate:
        """Estimate cost for a specific GPU configuration.

        Parameters
        ----------
        gpu_name:
            GPU model key from GPU_CATALOG. Defaults to instance config.
        num_gpus:
            Number of GPUs. Defaults to instance config.
        duration_hours:
            Duration in hours. Defaults to profiled duration.
        """
        name = gpu_name or self._gpu_name
        n = num_gpus if num_gpus is not None else self._num_gpus

        price = self._get_price(name)
        total_per_hour = price * n

        dur_h = duration_hours
        if dur_h is None and self._gpu_result is not None:
            dur_h = self._gpu_result.duration_s / 3600.0
        if dur_h is None:
            dur_h = 0.0

        session_cost = total_per_hour * dur_h

        util_pct = 0.0
        if self._gpu_result is not None:
            util_pct = self._gpu_result.avg_utilization

        effective_cost = total_per_hour / (util_pct / 100.0) if util_pct > 0 else total_per_hour

        catalog_entry = GPU_CATALOG.get(name, {})
        display_name = catalog_entry.get("name", name)

        return CostEstimate(
            gpu_name=display_name,
            num_gpus=n,
            cost_per_gpu_hour=round(price, 4),
            total_cost_per_hour=round(total_per_hour, 4),
            profiled_duration_hours=round(dur_h, 6),
            estimated_session_cost=round(session_cost, 4),
            gpu_utilization_pct=round(util_pct, 2),
            effective_cost_per_util_hour=round(effective_cost, 4),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_current_cost(self) -> CostEstimate:
        """Estimate cost for the current configuration."""
        return self.estimate_cost()

    def _recommend_cheaper_configs(
        self,
        current: CostEstimate,
    ) -> List[GPURecommendation]:
        """Find cheaper GPU configurations that could run the workload."""
        recommendations: List[GPURecommendation] = []

        required_memory_mb = self._get_required_memory_mb()
        required_memory_gb = required_memory_mb / 1024.0 if required_memory_mb > 0 else 0.0

        current_total = current.total_cost_per_hour
        if current_total <= 0:
            return recommendations

        for gpu_key, info in GPU_CATALOG.items():
            if gpu_key == self._gpu_name:
                continue

            mem_gb = info["memory_gb"]
            price = info["price_per_hour"]
            tflops = info["fp16_tflops"]

            # Check if GPU has enough memory.
            if required_memory_gb > 0 and mem_gb < required_memory_gb:
                continue

            # Estimate how many of these GPUs we'd need.
            # Simple heuristic: scale by FLOPS ratio.
            current_info = GPU_CATALOG.get(self._gpu_name, {})
            current_tflops = current_info.get("fp16_tflops", 312.0)

            if tflops > 0 and current_tflops > 0:
                # If the alternative GPU is slower, we might need more.
                flops_ratio = current_tflops / tflops
                needed_gpus = max(int(self._num_gpus * flops_ratio + 0.5), 1)
            else:
                needed_gpus = self._num_gpus

            alt_total = price * needed_gpus
            savings_pct = ((current_total - alt_total) / current_total) * 100.0

            if savings_pct <= 0:
                continue

            reason = self._build_reason(
                info, needed_gpus, savings_pct, current,
            )

            recommendations.append(GPURecommendation(
                gpu_key=gpu_key,
                gpu_name=info["name"],
                cost_per_gpu_hour=price,
                estimated_cost_per_hour=round(alt_total, 4),
                num_gpus=needed_gpus,
                memory_gb=mem_gb,
                fp16_tflops=tflops,
                estimated_savings_pct=round(savings_pct, 1),
                reason=reason,
            ))

        recommendations.sort(key=lambda r: r.estimated_savings_pct, reverse=True)
        return recommendations

    def _get_required_memory_mb(self) -> float:
        """Determine memory requirement from profiling data."""
        if self._peak_memory_mb is not None:
            return self._peak_memory_mb

        if self._gpu_result is not None and self._gpu_result.peak_memory_mb > 0:
            return self._gpu_result.peak_memory_mb

        return 0.0

    def _get_price(self, gpu_name: str) -> float:
        """Get the price per GPU per hour."""
        if self._custom_cost is not None and gpu_name == self._gpu_name:
            return self._custom_cost

        entry = GPU_CATALOG.get(gpu_name)
        if entry is not None:
            return float(entry["price_per_hour"])

        return 0.0

    @staticmethod
    def _build_reason(
        info: Dict[str, Any],
        needed_gpus: int,
        savings_pct: float,
        current: CostEstimate,
    ) -> str:
        """Build a human-readable reason for the recommendation."""
        name = info["name"]
        mem = info["memory_gb"]
        tflops = info["fp16_tflops"]
        provider = info.get("provider", "")

        parts = [f"{name} ({mem}GB, {tflops} FP16 TFLOPS)"]
        if needed_gpus > 1:
            parts.append(f"using {needed_gpus} GPU(s)")
        parts.append(f"saves ~{savings_pct:.0f}%")
        if provider:
            parts.append(f"available on {provider}")

        return " - ".join(parts)

    @staticmethod
    def _estimate_monthly_savings(
        current: CostEstimate,
        recommendations: List[GPURecommendation],
    ) -> float:
        """Estimate monthly savings from the best recommendation.

        Assumes 24/7 operation (730 hours/month).
        """
        if not recommendations:
            return 0.0

        best = recommendations[0]
        hourly_savings = current.total_cost_per_hour - best.estimated_cost_per_hour
        return hourly_savings * 730.0  # ~hours per month

    @staticmethod
    def _generate_summary(
        current: CostEstimate,
        recommendations: List[GPURecommendation],
        monthly_savings: float,
    ) -> str:
        """Generate a summary string."""
        parts: List[str] = [
            f"Current config: {current.num_gpus}x {current.gpu_name} "
            f"at ${current.total_cost_per_hour:.2f}/hr "
            f"({current.gpu_utilization_pct:.0f}% avg utilization)."
        ]

        if recommendations:
            best = recommendations[0]
            parts.append(
                f"Best alternative: {best.num_gpus}x {best.gpu_name} "
                f"at ${best.estimated_cost_per_hour:.2f}/hr "
                f"(saves {best.estimated_savings_pct:.0f}%)."
            )
            if monthly_savings > 0:
                parts.append(
                    f"Potential monthly savings: ${monthly_savings:,.0f}."
                )
        else:
            parts.append("No cheaper alternatives found for this workload.")

        return " ".join(parts)
