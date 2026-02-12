"""Automatic bottleneck identification engine.

Consumes profiling results from the GPU profiler, kernel tracer, memory
analyzer, communication profiler, and data-loading profiler, then determines
where training time is being spent, ranks identified bottlenecks by impact,
and produces actionable recommendations with expected speedup estimates.

The analyzer is designed to work with partial data -- e.g. when only GPU
metrics are available and kernel tracing was not enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTraceResult
from src.profiler.memory_analyzer import MemoryAnalysisResult
from src.profiler.communication_profiler import CommProfileResult
from src.profiler.data_loading_profiler import DataLoadingResult

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class TimeCategory(str, Enum):
    """High-level categories of where training time is spent."""

    compute = "compute"
    memory = "memory"
    communication = "communication"
    data_loading = "data_loading"
    idle = "idle"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class Bottleneck:
    """A single identified performance bottleneck."""

    category: TimeCategory
    time_pct: float
    description: str
    impact_score: float
    recommendations: list[str]
    expected_speedup: str

    def __post_init__(self) -> None:
        self.impact_score = max(0.0, min(1.0, self.impact_score))

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Bottleneck:
        raw_recommendations = data.get("recommendations", [])
        recommendations: list[str] = (
            [str(r) for r in raw_recommendations]
            if isinstance(raw_recommendations, list)
            else []
        )
        return cls(
            category=TimeCategory(data.get("category", "compute")),
            time_pct=float(data.get("time_pct", 0.0)),
            description=str(data.get("description", "")),
            impact_score=float(data.get("impact_score", 0.0)),
            recommendations=recommendations,
            expected_speedup=str(data.get("expected_speedup", "")),
        )


@dataclass
class PerformanceSummary:
    """Overall performance analysis of a training run."""

    total_time_s: float
    compute_pct: float
    memory_pct: float
    communication_pct: float
    data_loading_pct: float
    idle_pct: float
    primary_bottleneck: str
    bottlenecks: list[Bottleneck]
    overall_efficiency: float
    summary_text: str

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_time_s": self.total_time_s,
            "compute_pct": self.compute_pct,
            "memory_pct": self.memory_pct,
            "communication_pct": self.communication_pct,
            "data_loading_pct": self.data_loading_pct,
            "idle_pct": self.idle_pct,
            "primary_bottleneck": self.primary_bottleneck,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "overall_efficiency": self.overall_efficiency,
            "summary_text": self.summary_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceSummary:
        raw_bottlenecks = data.get("bottlenecks", [])
        bottlenecks: list[Bottleneck] = (
            [Bottleneck.from_dict(b) for b in raw_bottlenecks]
            if isinstance(raw_bottlenecks, list)
            else []
        )
        return cls(
            total_time_s=float(data.get("total_time_s", 0.0)),
            compute_pct=float(data.get("compute_pct", 0.0)),
            memory_pct=float(data.get("memory_pct", 0.0)),
            communication_pct=float(data.get("communication_pct", 0.0)),
            data_loading_pct=float(data.get("data_loading_pct", 0.0)),
            idle_pct=float(data.get("idle_pct", 0.0)),
            primary_bottleneck=str(data.get("primary_bottleneck", "")),
            bottlenecks=bottlenecks,
            overall_efficiency=float(data.get("overall_efficiency", 0.0)),
            summary_text=str(data.get("summary_text", "")),
        )


# ============================================================================
# Thresholds and constants
# ============================================================================

# SM utilization below this percentage is considered low.
_LOW_SM_UTILIZATION_PCT: float = 50.0

# SM utilization below this percentage is considered very low.
_VERY_LOW_SM_UTILIZATION_PCT: float = 30.0

# Memory bandwidth utilization above this percentage suggests memory-bound work.
_HIGH_MEM_BANDWIDTH_PCT: float = 70.0

# GPU memory usage above this percentage signals memory pressure.
_HIGH_MEMORY_USAGE_PCT: float = 80.0

# Fraction of kernel time spent on memory ops to flag as excessive.
_EXCESSIVE_MEMCPY_FRACTION: float = 0.15

# Communication time above this fraction of total time is a bottleneck.
_HIGH_COMM_FRACTION: float = 0.20

# Data loading idle fraction above this is considered a bottleneck.
_HIGH_DATA_LOADING_IDLE_FRACTION: float = 0.15

# Idle time above this fraction of total time is flagged.
_HIGH_IDLE_FRACTION: float = 0.10

# Theoretical peak FLOPS for common GPU architectures (FP16 TFLOPS).
_THEORETICAL_PEAK_FLOPS: Dict[str, float] = {
    "a100": 312.0,      # A100 SXM FP16 Tensor Core
    "h100": 989.5,      # H100 SXM FP16 Tensor Core
    "a10g": 125.0,      # A10G FP16 Tensor Core
    "v100": 125.0,      # V100 SXM2 FP16 Tensor Core
    "l40s": 362.0,      # L40S FP16 Tensor Core
    "4090": 330.0,      # RTX 4090 FP16 Tensor Core
    "3090": 142.0,      # RTX 3090 FP16 Tensor Core
}


# ============================================================================
# Bottleneck Detector
# ============================================================================


class BottleneckDetector:
    """Identify and rank performance bottlenecks from profiling results.

    Accepts optional results from each of the five profilers.  Any combination
    of results can be provided -- the detector adapts its analysis to the data
    that is available.

    Usage::

        detector = BottleneckDetector(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
        )
        summary = detector.analyze()
        for b in summary.bottlenecks:
            print(b.category, b.impact_score, b.recommendations)
    """

    def __init__(
        self,
        gpu_result: Optional[GPUProfileResult] = None,
        kernel_result: Optional[KernelTraceResult] = None,
        memory_result: Optional[MemoryAnalysisResult] = None,
        comm_result: Optional[CommProfileResult] = None,
        data_result: Optional[DataLoadingResult] = None,
    ) -> None:
        self._gpu_result = gpu_result
        self._kernel_result = kernel_result
        self._memory_result = memory_result
        self._comm_result = comm_result
        self._data_result = data_result

        self._bottlenecks: list[Bottleneck] = []
        self._time_breakdown: Dict[TimeCategory, float] = {
            cat: 0.0 for cat in TimeCategory
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> PerformanceSummary:
        """Run the full bottleneck analysis pipeline.

        Returns
        -------
        PerformanceSummary
            Aggregated summary with time breakdown, ranked bottlenecks,
            and an overall efficiency score.
        """
        self._bottlenecks = []

        # Step 1: categorize time.
        self._categorize_time()

        # Step 2: detect bottlenecks per category.
        self._detect_compute_bottlenecks()
        self._detect_memory_bottlenecks()
        self._detect_communication_bottlenecks()
        self._detect_data_loading_bottlenecks()
        self._detect_idle_bottlenecks()

        # Step 3: rank by impact.
        self._rank_bottlenecks()

        # Step 4: determine primary bottleneck.
        primary = self._determine_primary_bottleneck()

        # Step 5: compute overall efficiency.
        efficiency = self._compute_overall_efficiency()

        # Step 6: estimate theoretical performance.
        theoretical_info = self._estimate_theoretical_performance()
        if theoretical_info is not None:
            logger.debug(
                "Theoretical performance estimate: %.1f%% of peak.",
                theoretical_info * 100.0,
            )

        # Step 7: generate human-readable summary.
        total_time = self._estimate_total_time()
        summary_text = self._generate_summary_text(primary, efficiency)

        return PerformanceSummary(
            total_time_s=round(total_time, 3),
            compute_pct=round(self._time_breakdown[TimeCategory.compute], 1),
            memory_pct=round(self._time_breakdown[TimeCategory.memory], 1),
            communication_pct=round(self._time_breakdown[TimeCategory.communication], 1),
            data_loading_pct=round(self._time_breakdown[TimeCategory.data_loading], 1),
            idle_pct=round(self._time_breakdown[TimeCategory.idle], 1),
            primary_bottleneck=primary,
            bottlenecks=list(self._bottlenecks),
            overall_efficiency=round(efficiency, 3),
            summary_text=summary_text,
        )

    # ------------------------------------------------------------------
    # Time categorisation
    # ------------------------------------------------------------------

    def _categorize_time(self) -> None:
        """Break down total profiled time into the five categories.

        Uses whatever data is available.  When kernel-level tracing is
        present, the category breakdown comes from kernel durations.  When
        only GPU-level metrics are available, heuristics are applied based
        on SM activity and memory bandwidth utilization.
        """
        # Start from kernel-level data if available -- it provides the most
        # granular breakdown.
        if self._kernel_result is not None and self._kernel_result.total_kernel_time_us > 0:
            self._categorize_from_kernels()
        elif self._gpu_result is not None and self._gpu_result.snapshots:
            self._categorize_from_gpu_metrics()
        else:
            # Fallback: assign all time to compute (we have no data to split).
            self._time_breakdown[TimeCategory.compute] = 100.0
            logger.debug(
                "No GPU or kernel data available; defaulting to 100%% compute."
            )

        # Overlay communication time if comm data is available.
        if self._comm_result is not None and self._comm_result.total_comm_time_us > 0:
            self._overlay_communication_time()

        # Overlay data-loading time if data profiling is available.
        if self._data_result is not None and self._data_result.avg_gpu_idle_time_ms > 0:
            self._overlay_data_loading_time()

        # Normalise so percentages sum to 100.
        self._normalise_breakdown()

    def _categorize_from_kernels(self) -> None:
        """Derive the time breakdown from kernel-level category data."""
        assert self._kernel_result is not None
        breakdown = self._kernel_result.category_breakdown
        total_us = self._kernel_result.total_kernel_time_us
        if total_us <= 0:
            return

        # Map kernel categories to TimeCategory.
        compute_us = (
            breakdown.get("gemm", 0.0)
            + breakdown.get("conv", 0.0)
            + breakdown.get("attention", 0.0)
            + breakdown.get("elementwise", 0.0)
            + breakdown.get("reduction", 0.0)
            + breakdown.get("other", 0.0)
        )
        memory_us = breakdown.get("memory", 0.0)
        comm_us = breakdown.get("communication", 0.0)

        self._time_breakdown[TimeCategory.compute] = (compute_us / total_us) * 100.0
        self._time_breakdown[TimeCategory.memory] = (memory_us / total_us) * 100.0
        self._time_breakdown[TimeCategory.communication] = (comm_us / total_us) * 100.0

        logger.debug(
            "Kernel-based breakdown: compute=%.1f%%, memory=%.1f%%, communication=%.1f%%.",
            self._time_breakdown[TimeCategory.compute],
            self._time_breakdown[TimeCategory.memory],
            self._time_breakdown[TimeCategory.communication],
        )

    def _categorize_from_gpu_metrics(self) -> None:
        """Derive the time breakdown from GPU-level metrics using heuristics.

        When only GPU utilization and memory bandwidth data are available we
        estimate the split as follows:
          - High SM activity + low memory bandwidth  -> compute-bound
          - Low SM activity + high memory bandwidth   -> memory-bound
          - Low SM activity + low memory bandwidth    -> idle / other
        """
        assert self._gpu_result is not None
        snapshots = self._gpu_result.snapshots
        if not snapshots:
            return

        total = len(snapshots)
        compute_count = 0
        memory_count = 0
        idle_count = 0

        for snap in snapshots:
            sm = snap.sm_activity_pct
            mem_bw = snap.memory_bandwidth_pct

            if sm >= _LOW_SM_UTILIZATION_PCT:
                compute_count += 1
            elif mem_bw >= _HIGH_MEM_BANDWIDTH_PCT:
                memory_count += 1
            elif sm < _VERY_LOW_SM_UTILIZATION_PCT and mem_bw < 20.0:
                idle_count += 1
            else:
                # Moderate SM and memory bandwidth -- attribute to compute.
                compute_count += 1

        self._time_breakdown[TimeCategory.compute] = (compute_count / total) * 100.0
        self._time_breakdown[TimeCategory.memory] = (memory_count / total) * 100.0
        self._time_breakdown[TimeCategory.idle] = (idle_count / total) * 100.0

        logger.debug(
            "GPU-metric heuristic breakdown: compute=%.1f%%, memory=%.1f%%, idle=%.1f%%.",
            self._time_breakdown[TimeCategory.compute],
            self._time_breakdown[TimeCategory.memory],
            self._time_breakdown[TimeCategory.idle],
        )

    def _overlay_communication_time(self) -> None:
        """Incorporate communication profiler data into the time breakdown.

        If the communication profiler recorded data, the non-overlapped
        communication fraction is added to the breakdown and the remaining
        categories are scaled down proportionally.
        """
        assert self._comm_result is not None
        total_time_us = self._estimate_total_time() * 1e6
        if total_time_us <= 0:
            return

        overlap_pct = self._comm_result.comm_compute_overlap_pct
        non_overlapped_fraction = 1.0 - (overlap_pct / 100.0)
        comm_time_us = self._comm_result.total_comm_time_us * non_overlapped_fraction
        comm_pct = min((comm_time_us / total_time_us) * 100.0, 100.0)

        if comm_pct > 0:
            # Scale down existing categories to make room for communication.
            existing_total = sum(self._time_breakdown.values())
            if existing_total > 0:
                scale = max(100.0 - comm_pct, 0.0) / existing_total
                for cat in TimeCategory:
                    if cat != TimeCategory.communication:
                        self._time_breakdown[cat] *= scale
            self._time_breakdown[TimeCategory.communication] = comm_pct

    def _overlay_data_loading_time(self) -> None:
        """Incorporate data-loading profiler data into the time breakdown.

        Uses the average GPU idle time attributable to data loading.
        """
        assert self._data_result is not None
        if not self._data_result.metrics:
            return

        # Estimate fraction of time spent waiting for data.
        avg_idle_ms = self._data_result.avg_gpu_idle_time_ms
        avg_load_ms = self._data_result.avg_load_time_ms

        # Use the larger of idle and load time as the data loading penalty.
        data_penalty_ms = max(avg_idle_ms, avg_load_ms)
        if data_penalty_ms <= 0:
            return

        # Estimate total step time from data loading metrics.
        metrics = self._data_result.metrics
        avg_total_ms = sum(
            m.batch_load_time_ms + m.batch_process_time_ms for m in metrics
        ) / len(metrics)

        if avg_total_ms <= 0:
            return

        data_pct = min((data_penalty_ms / avg_total_ms) * 100.0, 100.0)

        if data_pct > 0:
            existing_total = sum(self._time_breakdown.values())
            if existing_total > 0:
                scale = max(100.0 - data_pct, 0.0) / existing_total
                for cat in TimeCategory:
                    if cat != TimeCategory.data_loading:
                        self._time_breakdown[cat] *= scale
            self._time_breakdown[TimeCategory.data_loading] = data_pct

    def _normalise_breakdown(self) -> None:
        """Ensure the time breakdown sums to exactly 100%."""
        total = sum(self._time_breakdown.values())
        if total <= 0:
            # No data at all; default to 100% compute.
            self._time_breakdown[TimeCategory.compute] = 100.0
            return

        if abs(total - 100.0) > 0.1:
            factor = 100.0 / total
            for cat in TimeCategory:
                self._time_breakdown[cat] *= factor

    # ------------------------------------------------------------------
    # Bottleneck detection: compute
    # ------------------------------------------------------------------

    def _detect_compute_bottlenecks(self) -> None:
        """Check for compute-related bottlenecks.

        Looks for:
        - Low SM utilization
        - Suboptimal kernels (low occupancy)
        - No TF32 or tensor core usage indicators
        - High fraction of non-GEMM compute (elementwise/reduction heavy)
        """
        compute_pct = self._time_breakdown[TimeCategory.compute]

        # -- Low SM utilization -----------------------------------------------
        if self._gpu_result is not None and self._gpu_result.snapshots:
            snapshots = self._gpu_result.snapshots
            avg_sm = sum(s.sm_activity_pct for s in snapshots) / len(snapshots)

            if avg_sm < _VERY_LOW_SM_UTILIZATION_PCT:
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.compute,
                    time_pct=compute_pct,
                    description=(
                        f"Very low SM utilization ({avg_sm:.1f}%). The GPU "
                        f"compute units are largely idle during training steps."
                    ),
                    impact_score=0.9,
                    recommendations=[
                        "Increase batch size to improve GPU utilization and amortize kernel launch overhead.",
                        "Enable TF32: add torch.backends.cuda.matmul.allow_tf32 = True "
                        "and torch.backends.cudnn.allow_tf32 = True at the start of your script.",
                        "Check for excessive CPU-GPU synchronization points "
                        "(e.g. .item(), .cpu(), print(tensor)) that serialize execution.",
                    ],
                    expected_speedup="2.0-4.0x",
                ))
            elif avg_sm < _LOW_SM_UTILIZATION_PCT:
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.compute,
                    time_pct=compute_pct,
                    description=(
                        f"Low SM utilization ({avg_sm:.1f}%). GPU compute "
                        f"resources are underutilized."
                    ),
                    impact_score=0.6,
                    recommendations=[
                        "Increase batch size to better saturate GPU SMs.",
                        "Enable TF32: add torch.backends.cuda.matmul.allow_tf32 = True "
                        "and torch.backends.cudnn.allow_tf32 = True at the start of your script.",
                        "Use torch.compile() to fuse operations and reduce kernel launch overhead.",
                    ],
                    expected_speedup="1.3-1.8x",
                ))

        # -- Suboptimal kernels (low occupancy) -------------------------------
        if self._kernel_result is not None and self._kernel_result.inefficient_kernels:
            inefficient = self._kernel_result.inefficient_kernels
            total_us = self._kernel_result.total_kernel_time_us
            if total_us > 0:
                inefficient_time = sum(k.duration_us for k in inefficient)
                inefficient_pct = (inefficient_time / total_us) * 100.0

                if inefficient_pct > 10.0:
                    top_names = [k.name for k in sorted(
                        inefficient, key=lambda k: k.duration_us, reverse=True
                    )[:3]]
                    names_str = ", ".join(top_names)

                    self._bottlenecks.append(Bottleneck(
                        category=TimeCategory.compute,
                        time_pct=inefficient_pct,
                        description=(
                            f"Inefficient kernels consume {inefficient_pct:.1f}% of "
                            f"GPU time. Top offenders: {names_str}."
                        ),
                        impact_score=min(inefficient_pct / 100.0 + 0.2, 1.0),
                        recommendations=[
                            "Use torch.compile() with mode='max-autotune' to let the "
                            "compiler find faster kernel implementations.",
                            "Enable flash attention: install flash-attn package "
                            "(pip install flash-attn) and use "
                            "torch.nn.functional.scaled_dot_product_attention().",
                            "Consider using CUDA graphs to eliminate kernel launch overhead "
                            "for repeated execution patterns: torch.cuda.CUDAGraph().",
                        ],
                        expected_speedup="1.2-1.5x",
                    ))

        # -- Heavy elementwise / reduction workload ---------------------------
        if self._kernel_result is not None and self._kernel_result.category_breakdown:
            breakdown = self._kernel_result.category_breakdown
            total_us = self._kernel_result.total_kernel_time_us
            if total_us > 0:
                gemm_us = breakdown.get("gemm", 0.0) + breakdown.get("conv", 0.0)
                elementwise_us = breakdown.get("elementwise", 0.0) + breakdown.get("reduction", 0.0)
                gemm_pct = (gemm_us / total_us) * 100.0
                elementwise_pct = (elementwise_us / total_us) * 100.0

                # If elementwise + reduction dominates over GEMM, the workload
                # is not well-suited for tensor cores.
                if elementwise_pct > 40.0 and gemm_pct < 30.0:
                    self._bottlenecks.append(Bottleneck(
                        category=TimeCategory.compute,
                        time_pct=elementwise_pct,
                        description=(
                            f"Elementwise and reduction kernels dominate "
                            f"({elementwise_pct:.1f}% of kernel time vs "
                            f"{gemm_pct:.1f}% GEMM). Tensor cores are underutilized."
                        ),
                        impact_score=0.5,
                        recommendations=[
                            "Use torch.compile() to fuse elementwise operations "
                            "into fewer, more efficient kernels.",
                            "Replace manual normalization/activation sequences with "
                            "fused implementations (e.g. apex.normalization.FusedLayerNorm, "
                            "xformers.ops.fused_linear_layer).",
                            "Ensure model dimensions (hidden size, intermediate size) "
                            "are multiples of 8 for optimal tensor core alignment.",
                        ],
                        expected_speedup="1.1-1.4x",
                    ))

                # Low GEMM percentage with attention present.
                attention_us = breakdown.get("attention", 0.0)
                attention_pct = (attention_us / total_us) * 100.0
                if attention_pct > 20.0 and "flash" not in str(
                    [k.name.lower() for k in self._kernel_result.kernels
                     if k.category == "attention"]
                ):
                    self._bottlenecks.append(Bottleneck(
                        category=TimeCategory.compute,
                        time_pct=attention_pct,
                        description=(
                            f"Attention kernels consume {attention_pct:.1f}% of "
                            f"kernel time and may not be using flash attention."
                        ),
                        impact_score=0.7,
                        recommendations=[
                            "Enable flash attention: install flash-attn package "
                            "(pip install flash-attn --no-build-isolation) for "
                            "2-4x faster attention with O(N) memory.",
                            "Use torch.nn.functional.scaled_dot_product_attention() "
                            "which automatically selects the best attention backend "
                            "(FlashAttention, Memory-Efficient, or Math).",
                            "For very long sequences, consider ring attention or "
                            "blockwise parallel attention implementations.",
                        ],
                        expected_speedup="1.5-2.5x",
                    ))

    # ------------------------------------------------------------------
    # Bottleneck detection: memory
    # ------------------------------------------------------------------

    def _detect_memory_bottlenecks(self) -> None:
        """Check for memory-related bottlenecks.

        Looks for:
        - High memory bandwidth utilization (memory-bound kernels)
        - High GPU memory pressure / near-OOM conditions
        - Excessive memcpy/memset operations
        - Memory leaks
        - Frequent small allocations
        """
        memory_pct = self._time_breakdown[TimeCategory.memory]

        # -- High memory bandwidth utilization --------------------------------
        if self._gpu_result is not None and self._gpu_result.snapshots:
            snapshots = self._gpu_result.snapshots
            avg_mem_bw = sum(s.memory_bandwidth_pct for s in snapshots) / len(snapshots)

            if avg_mem_bw > _HIGH_MEM_BANDWIDTH_PCT:
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.memory,
                    time_pct=memory_pct,
                    description=(
                        f"High memory bandwidth utilization ({avg_mem_bw:.1f}%). "
                        f"Kernels are likely memory-bound, spending more time "
                        f"moving data than computing."
                    ),
                    impact_score=0.7,
                    recommendations=[
                        "Use mixed precision training (torch.cuda.amp.autocast with "
                        "dtype=torch.bfloat16) to halve memory traffic for activations "
                        "and gradients.",
                        "Enable TF32: add torch.backends.cuda.matmul.allow_tf32 = True "
                        "to use TF32 precision which reduces memory bandwidth requirements "
                        "for matmul operations.",
                        "Use torch.compile() to fuse memory-bound elementwise ops into "
                        "single kernels, reducing round-trips to global memory.",
                    ],
                    expected_speedup="1.3-1.8x",
                ))

        # -- Excessive memcpy traffic -----------------------------------------
        if self._kernel_result is not None and self._kernel_result.category_breakdown:
            breakdown = self._kernel_result.category_breakdown
            total_us = self._kernel_result.total_kernel_time_us
            if total_us > 0:
                memcpy_us = breakdown.get("memory", 0.0)
                memcpy_fraction = memcpy_us / total_us

                if memcpy_fraction > _EXCESSIVE_MEMCPY_FRACTION:
                    memcpy_pct = memcpy_fraction * 100.0
                    self._bottlenecks.append(Bottleneck(
                        category=TimeCategory.memory,
                        time_pct=memcpy_pct,
                        description=(
                            f"Excessive memory copy operations ({memcpy_pct:.1f}% of "
                            f"kernel time). Data is being moved unnecessarily between "
                            f"host and device or between memory locations."
                        ),
                        impact_score=min(memcpy_fraction + 0.3, 1.0),
                        recommendations=[
                            "Enable pin_memory=True on DataLoader to use page-locked "
                            "memory for faster host-to-device transfers.",
                            "Avoid unnecessary .cpu() / .cuda() / .to(device) calls "
                            "in the training loop; keep tensors on-device.",
                            "Use non_blocking=True for .to(device) calls to overlap "
                            "transfers with computation: tensor.to(device, non_blocking=True).",
                            "Pre-allocate output buffers and use in-place operations to "
                            "reduce temporary allocations.",
                        ],
                        expected_speedup="1.1-1.3x",
                    ))

        # -- High GPU memory usage / OOM risk ---------------------------------
        if self._memory_result is not None:
            oom_risk = self._memory_result.oom_risk
            if oom_risk.peak_usage_pct > _HIGH_MEMORY_USAGE_PCT:
                risk_desc = "critical" if oom_risk.peak_usage_pct > 90.0 else "high"
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.memory,
                    time_pct=memory_pct,
                    description=(
                        f"GPU memory usage is {risk_desc} ({oom_risk.peak_usage_pct:.1f}% "
                        f"of total, {oom_risk.headroom_mb:.0f} MB headroom). "
                        f"This limits batch size and may cause OOM errors."
                    ),
                    impact_score=min(oom_risk.risk_score + 0.2, 1.0),
                    recommendations=[
                        "Enable gradient checkpointing: model.gradient_checkpointing_enable() "
                        "to trade ~30% compute for ~60% activation memory savings.",
                        "Use mixed precision training: torch.cuda.amp.autocast(dtype=torch.bfloat16) "
                        "to halve activation and gradient memory.",
                        "Use a memory-efficient optimizer: pip install bitsandbytes, then "
                        "import bitsandbytes as bnb; optimizer = bnb.optim.Adam8bit(params, lr=1e-4).",
                        "If using multi-GPU, enable ZeRO stage 2+ to shard optimizer "
                        "states across GPUs: deepspeed.initialize(config={'zero_optimization': "
                        "{'stage': 2}}).",
                    ],
                    expected_speedup="1.2-2.0x (via larger batch size)",
                ))

            # -- Memory leak ---------------------------------------------------
            if self._memory_result.leak_detected:
                num_leaks = len(self._memory_result.leak_points)
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.memory,
                    time_pct=0.0,
                    description=(
                        f"Memory leak detected ({num_leaks} leak point(s)). "
                        f"GPU memory is growing monotonically, which will "
                        f"eventually cause an OOM error."
                    ),
                    impact_score=0.8,
                    recommendations=[
                        "Ensure loss.backward() is called on a scalar loss, not an "
                        "accumulated tensor. Use loss.item() when logging: "
                        "log_value = loss.item().",
                        "Check that the computation graph is not being retained "
                        "across iterations. Avoid storing tensors that require grad "
                        "in lists or dictionaries that persist across steps.",
                        "Call torch.cuda.empty_cache() periodically to release "
                        "unused cached memory back to the allocator.",
                        "Use the PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                        "environment variable to reduce memory fragmentation.",
                    ],
                    expected_speedup="prevents OOM crash",
                ))

        # -- Breakdown-specific: large optimizer state ------------------------
        if (
            self._memory_result is not None
            and self._memory_result.breakdown is not None
        ):
            bd = self._memory_result.breakdown
            total_known = (
                bd.parameter_memory_mb
                + bd.gradient_memory_mb
                + bd.optimizer_state_mb
                + bd.activation_memory_mb
                + bd.other_mb
            )
            if total_known > 0 and bd.optimizer_state_mb > 2.0 * bd.parameter_memory_mb:
                opt_pct = (bd.optimizer_state_mb / total_known) * 100.0
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.memory,
                    time_pct=0.0,
                    description=(
                        f"Optimizer states consume {opt_pct:.0f}% of allocated "
                        f"GPU memory ({bd.optimizer_state_mb:.0f} MB), which is "
                        f"over 2x the parameter memory ({bd.parameter_memory_mb:.0f} MB)."
                    ),
                    impact_score=0.5,
                    recommendations=[
                        "Switch to a memory-efficient optimizer: pip install bitsandbytes, "
                        "then use bnb.optim.Adam8bit(params, lr=1e-4) to reduce optimizer "
                        "state memory by 75%.",
                        "Use Adafactor instead of Adam to eliminate first-moment state: "
                        "from transformers.optimization import Adafactor.",
                        "For multi-GPU training, enable ZeRO stage 1 to shard optimizer "
                        "states across GPUs.",
                    ],
                    expected_speedup="1.2-1.5x (via larger batch size)",
                ))

    # ------------------------------------------------------------------
    # Bottleneck detection: communication
    # ------------------------------------------------------------------

    def _detect_communication_bottlenecks(self) -> None:
        """Check for communication-related bottlenecks.

        Looks for:
        - High communication overhead (large fraction of total time)
        - Poor communication/compute overlap
        - Straggler ranks
        - Low bandwidth utilization
        """
        comm_pct = self._time_breakdown[TimeCategory.communication]

        if self._comm_result is None or self._comm_result.total_comm_time_us <= 0:
            return

        total_time_us = self._estimate_total_time() * 1e6
        if total_time_us <= 0:
            return

        comm_fraction = self._comm_result.total_comm_time_us / total_time_us

        # -- High communication overhead --------------------------------------
        if comm_fraction > _HIGH_COMM_FRACTION:
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.communication,
                time_pct=comm_pct,
                description=(
                    f"Communication overhead is {comm_pct:.1f}% of total time. "
                    f"Inter-GPU data transfer is consuming a significant portion "
                    f"of each training step."
                ),
                impact_score=min(comm_fraction + 0.2, 1.0),
                recommendations=[
                    "Enable gradient bucketing with a larger bucket size: "
                    "torch.nn.parallel.DistributedDataParallel(model, "
                    "bucket_cap_mb=50) to amortize communication overhead.",
                    "Use gradient compression: install torch-distributed-fsdp or "
                    "PowerSGD (torch.distributed.algorithms.ddp_comm_hooks."
                    "powerSGD_hook.powerSGD_hook).",
                    "Overlap communication with computation by enabling "
                    "gradient_as_bucket_view=True in DDP.",
                    "For large models, switch from DDP to FSDP "
                    "(torch.distributed.fsdp.FullyShardedDataParallel) to shard "
                    "parameters, gradients, and optimizer states across GPUs.",
                ],
                expected_speedup="1.2-1.8x",
            ))

        # -- Poor communication/compute overlap -------------------------------
        overlap_pct = self._comm_result.comm_compute_overlap_pct
        if overlap_pct < 30.0 and comm_fraction > 0.10:
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.communication,
                time_pct=comm_pct * (1.0 - overlap_pct / 100.0),
                description=(
                    f"Communication/compute overlap is only {overlap_pct:.1f}%. "
                    f"The GPU is idle while waiting for collective operations "
                    f"to complete."
                ),
                impact_score=0.6,
                recommendations=[
                    "Enable overlap_with_ddp=True in DDP or use FSDP with "
                    "forward_prefetch=True and backward_prefetch="
                    "BackwardPrefetch.BACKWARD_PRE.",
                    "Use torch.distributed.algorithms.ddp_comm_hooks."
                    "default_hooks.fp16_compress_hook to reduce bytes transferred "
                    "and speed up collective operations.",
                    "Increase the number of gradient bucketing buckets to enable "
                    "earlier communication: DDP(model, bucket_cap_mb=25).",
                ],
                expected_speedup="1.1-1.4x",
            ))

        # -- Straggler ranks --------------------------------------------------
        if self._comm_result.straggler_ranks:
            stragglers = self._comm_result.straggler_ranks
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.communication,
                time_pct=comm_pct,
                description=(
                    f"Straggler GPU rank(s) detected: {stragglers}. These "
                    f"ranks are significantly slower in collective operations, "
                    f"causing all other ranks to wait."
                ),
                impact_score=0.7,
                recommendations=[
                    "Check for thermal throttling on straggler GPUs using "
                    "nvidia-smi -q -d PERFORMANCE.",
                    "Verify interconnect topology: straggler ranks may be on a "
                    "slower PCIe link instead of NVLink. Check with nvidia-smi topo -m.",
                    "Ensure workload is evenly distributed across ranks. Check "
                    "for uneven batch sizes or sequence lengths.",
                    "Consider using join() context manager for uneven inputs: "
                    "with model.join(): train_step().",
                ],
                expected_speedup="1.1-1.3x",
            ))

        # -- Low bandwidth utilization ----------------------------------------
        if self._comm_result.avg_bandwidth_gbps > 0:
            # Compare against a rough NVLink estimate.
            # If bandwidth is below 50 Gbps, it's likely PCIe-bound.
            if self._comm_result.avg_bandwidth_gbps < 50.0 and comm_fraction > 0.10:
                self._bottlenecks.append(Bottleneck(
                    category=TimeCategory.communication,
                    time_pct=comm_pct,
                    description=(
                        f"Low communication bandwidth "
                        f"({self._comm_result.avg_bandwidth_gbps:.1f} Gbps). "
                        f"Collective operations may be running over PCIe instead "
                        f"of NVLink, or message sizes are too small for efficient "
                        f"transfers."
                    ),
                    impact_score=0.5,
                    recommendations=[
                        "Verify GPU interconnect topology with nvidia-smi topo -m. "
                        "Ensure GPUs communicating frequently are on the same NVLink "
                        "domain.",
                        "Increase gradient bucket size (DDP bucket_cap_mb=50+) to "
                        "send larger messages that amortize latency.",
                        "Set NCCL environment variables for tuning: "
                        "NCCL_ALGO=Ring, NCCL_MIN_NCHANNELS=4.",
                    ],
                    expected_speedup="1.1-1.3x",
                ))

    # ------------------------------------------------------------------
    # Bottleneck detection: data loading
    # ------------------------------------------------------------------

    def _detect_data_loading_bottlenecks(self) -> None:
        """Check for data pipeline bottlenecks.

        Looks for:
        - High GPU idle time due to data starvation
        - Low I/O throughput
        - Suboptimal DataLoader configuration
        """
        data_pct = self._time_breakdown[TimeCategory.data_loading]

        if self._data_result is None:
            return

        # -- Data loading is a bottleneck (profiler already classified) --------
        if self._data_result.is_bottleneck:
            severity = self._data_result.bottleneck_severity
            severity_scores = {
                "severe": 0.9,
                "moderate": 0.6,
                "mild": 0.4,
                "none": 0.1,
            }
            impact = severity_scores.get(severity, 0.3)

            recommendations: list[str] = []

            # Pull recommendations from the data loading profiler, and add
            # our own more specific ones.
            if self._data_result.recommendations:
                recommendations.extend(self._data_result.recommendations)

            # Add specific recommendations the data profiler may not have.
            recommendations.extend([
                "Use a high-performance data format: pip install webdataset; "
                "WebDataset provides sequential I/O patterns for maximum throughput.",
                "Consider FFCV for extreme data loading speed: pip install ffcv; "
                "FFCV can saturate NVMe bandwidth with minimal CPU overhead.",
            ])

            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.data_loading,
                time_pct=data_pct,
                description=(
                    f"Data loading is a {severity} bottleneck. The GPU spends "
                    f"{self._data_result.avg_gpu_idle_time_ms:.1f} ms idle per "
                    f"batch waiting for data."
                ),
                impact_score=impact,
                recommendations=recommendations,
                expected_speedup="1.2-2.0x",
            ))

        # -- Low I/O throughput -----------------------------------------------
        if self._data_result.io_throughput_mbps > 0 and self._data_result.io_throughput_mbps < 100.0:
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.data_loading,
                time_pct=data_pct,
                description=(
                    f"Low I/O throughput ({self._data_result.io_throughput_mbps:.1f} "
                    f"MB/s). Data is not being read from storage fast enough to "
                    f"keep the GPU fed."
                ),
                impact_score=0.5,
                recommendations=[
                    "Move training data to local NVMe SSD storage instead of "
                    "network-attached or spinning disk storage.",
                    "Use memory-mapped files or a RAM disk for datasets that fit "
                    "in system memory: mount -t tmpfs -o size=50G tmpfs /data.",
                    "Convert dataset to a sequential-read format (WebDataset, "
                    "TFRecord, FFCV) to minimize random I/O seeks.",
                ],
                expected_speedup="1.1-1.5x",
            ))

    # ------------------------------------------------------------------
    # Bottleneck detection: idle
    # ------------------------------------------------------------------

    def _detect_idle_bottlenecks(self) -> None:
        """Check for unexplained GPU idle time.

        Looks for:
        - High idle fraction not attributable to data loading or communication
        - CPU-GPU synchronization overhead
        """
        idle_pct = self._time_breakdown[TimeCategory.idle]

        if idle_pct < _HIGH_IDLE_FRACTION * 100.0:
            return

        # Check if the idle time is already explained by data loading or comm.
        data_pct = self._time_breakdown[TimeCategory.data_loading]
        comm_pct = self._time_breakdown[TimeCategory.communication]
        unexplained_idle = idle_pct - max(data_pct, comm_pct)

        if unexplained_idle > _HIGH_IDLE_FRACTION * 100.0:
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.idle,
                time_pct=idle_pct,
                description=(
                    f"GPU is idle {idle_pct:.1f}% of the time. This suggests "
                    f"CPU-bound preprocessing, synchronization barriers, or "
                    f"Python overhead in the training loop."
                ),
                impact_score=min(idle_pct / 100.0 + 0.2, 1.0),
                recommendations=[
                    "Remove unnecessary CPU-GPU synchronization: avoid .item(), "
                    ".numpy(), .cpu() in the training loop. Use torch.cuda.synchronize() "
                    "only when explicitly needed for timing.",
                    "Move preprocessing to GPU: use torchvision.transforms.v2 with "
                    "device='cuda' or NVIDIA DALI for GPU-accelerated data augmentation.",
                    "Use torch.compile() to reduce Python overhead by compiling "
                    "the model into optimized kernels: model = torch.compile(model).",
                    "Set CUDA_LAUNCH_BLOCKING=0 (default) and verify it is not set "
                    "to 1 in your environment, as that forces synchronous execution.",
                ],
                expected_speedup="1.2-2.0x",
            ))
        elif idle_pct > _HIGH_IDLE_FRACTION * 100.0:
            # Idle is mostly explained by known causes but still high.
            self._bottlenecks.append(Bottleneck(
                category=TimeCategory.idle,
                time_pct=idle_pct,
                description=(
                    f"GPU idle time is {idle_pct:.1f}%, partially attributable to "
                    f"data loading ({data_pct:.1f}%) and communication ({comm_pct:.1f}%)."
                ),
                impact_score=0.3,
                recommendations=[
                    "Address data loading and communication bottlenecks first; "
                    "idle time should decrease as a consequence.",
                    "Profile CPU utilization alongside GPU to identify CPU-side "
                    "bottlenecks (e.g. py-spy, cProfile).",
                ],
                expected_speedup="1.1-1.3x",
            ))

    # ------------------------------------------------------------------
    # Ranking and summary
    # ------------------------------------------------------------------

    def _rank_bottlenecks(self) -> None:
        """Sort bottlenecks by impact score (highest first)."""
        self._bottlenecks.sort(key=lambda b: b.impact_score, reverse=True)

    def _determine_primary_bottleneck(self) -> str:
        """Return the name of the primary bottleneck category.

        If bottlenecks were detected, this is the category of the
        highest-impact bottleneck.  Otherwise it is the category with the
        highest time percentage.
        """
        if self._bottlenecks:
            return self._bottlenecks[0].category.value

        # No specific bottlenecks found; return the dominant time category.
        return max(self._time_breakdown, key=self._time_breakdown.get).value  # type: ignore[arg-type]

    def _compute_overall_efficiency(self) -> float:
        """Compute a single efficiency score in [0, 1].

        The score represents how effectively the GPU is being utilized.
        A score of 1.0 means the GPU is fully utilized on useful compute
        work with no bottlenecks.
        """
        # Start with compute fraction as a base.
        compute_fraction = self._time_breakdown[TimeCategory.compute] / 100.0

        # Penalize for each detected bottleneck, weighted by impact.
        penalty = 0.0
        for b in self._bottlenecks:
            penalty += b.impact_score * 0.1

        efficiency = max(compute_fraction - penalty, 0.0)

        # Also factor in SM utilization if available.
        if self._gpu_result is not None and self._gpu_result.snapshots:
            snapshots = self._gpu_result.snapshots
            avg_sm = sum(s.sm_activity_pct for s in snapshots) / len(snapshots)
            sm_factor = avg_sm / 100.0
            # Blend: 60% time-based, 40% SM-based.
            efficiency = 0.6 * efficiency + 0.4 * sm_factor

        return max(0.0, min(1.0, efficiency))

    def _generate_summary_text(self, primary_bottleneck: str, efficiency: float) -> str:
        """Generate the human-readable summary string.

        Parameters
        ----------
        primary_bottleneck:
            Name of the primary bottleneck category.
        efficiency:
            Overall efficiency score in [0, 1].
        """
        compute_pct = self._time_breakdown[TimeCategory.compute]
        category_labels = {
            "compute": "compute-bound",
            "memory": "memory-bound",
            "communication": "communication-bound",
            "data_loading": "data loading-bound",
            "idle": "idle/synchronization-bound",
        }
        bound_label = category_labels.get(primary_bottleneck, primary_bottleneck)

        parts: list[str] = [
            f"Your training is {compute_pct:.0f}% {bound_label}."
        ]

        if self._bottlenecks:
            top = self._bottlenecks[0]
            if top.recommendations:
                # Grab the first actionable recommendation.
                first_rec = top.recommendations[0]
                # Truncate at the first period for brevity, if it's long.
                if len(first_rec) > 80:
                    # Find the first sentence or clause.
                    colon_idx = first_rec.find(":")
                    if 0 < colon_idx < 60:
                        first_rec = first_rec[: colon_idx + 1].rstrip()
                    else:
                        period_idx = first_rec.find(".")
                        if 0 < period_idx < 80:
                            first_rec = first_rec[: period_idx + 1]
                        else:
                            first_rec = first_rec[:77] + "..."

                parts.append(
                    f"Top recommendation: {first_rec}"
                )
                if top.expected_speedup:
                    parts.append(
                        f"Expected speedup: {top.expected_speedup}."
                    )

        parts.append(
            f"Overall GPU efficiency: {efficiency * 100:.0f}%."
        )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Theoretical performance estimation
    # ------------------------------------------------------------------

    def _estimate_theoretical_performance(self) -> Optional[float]:
        """Compare observed performance against theoretical peak FLOPS.

        Returns
        -------
        float | None
            Fraction of theoretical peak achieved (0-1), or ``None`` if
            insufficient data is available to make the estimate.
        """
        if self._gpu_result is None or not self._gpu_result.snapshots:
            return None

        # Use SM utilization as a rough proxy for FLOPS utilization.
        snapshots = self._gpu_result.snapshots
        avg_sm = sum(s.sm_activity_pct for s in snapshots) / len(snapshots)

        # SM utilization is an upper bound on compute utilization because
        # the SMs may be active but executing memory-bound kernels.  Apply
        # a correction factor based on the memory bandwidth utilization.
        avg_mem_bw = sum(s.memory_bandwidth_pct for s in snapshots) / len(snapshots)

        # If memory bandwidth is high, the effective compute fraction is
        # lower than what SM utilization alone suggests.
        if avg_mem_bw > 50.0:
            correction = 1.0 - (avg_mem_bw - 50.0) / 100.0
        else:
            correction = 1.0

        theoretical_fraction = (avg_sm / 100.0) * correction
        return max(0.0, min(1.0, theoretical_fraction))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_total_time(self) -> float:
        """Estimate the total profiled time in seconds.

        Uses the GPU profiler's duration if available, otherwise falls back
        to kernel time or communication time.
        """
        if self._gpu_result is not None and self._gpu_result.duration_s > 0:
            return self._gpu_result.duration_s

        if self._kernel_result is not None and self._kernel_result.total_kernel_time_us > 0:
            return self._kernel_result.total_kernel_time_us / 1e6

        if self._comm_result is not None and self._comm_result.total_comm_time_us > 0:
            return self._comm_result.total_comm_time_us / 1e6

        if self._data_result is not None and self._data_result.metrics:
            total_ms = sum(
                m.batch_load_time_ms + m.batch_process_time_ms
                for m in self._data_result.metrics
            )
            return total_ms / 1000.0

        return 0.0
