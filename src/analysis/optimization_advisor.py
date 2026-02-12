"""Optimization advisor for ML training workloads.

Analyzes profiling results and generates specific, actionable optimization
recommendations with concrete parameter values (batch sizes, data loader
workers, precision modes, etc.).

The advisor goes beyond generic advice by computing recommended values
from the actual profiling data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTraceResult
from src.profiler.memory_analyzer import MemoryAnalysisResult
from src.profiler.data_loading_profiler import DataLoadingResult
from src.profiler.distributed_profiler import DistributedProfileResult
from src.analysis.bottleneck_detector import BottleneckDetector, PerformanceSummary

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class Priority(str, Enum):
    """Optimization priority level."""

    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class OptimizationCategory(str, Enum):
    """Category of optimization."""

    batch_size = "batch_size"
    precision = "precision"
    gradient_checkpointing = "gradient_checkpointing"
    data_loading = "data_loading"
    compilation = "compilation"
    distributed = "distributed"
    memory = "memory"
    kernel = "kernel"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class Optimization:
    """A single optimization recommendation."""

    category: OptimizationCategory
    title: str
    description: str
    priority: Priority
    estimated_speedup: str
    code_change: str  # concrete code snippet
    current_value: str
    recommended_value: str
    confidence: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optimization:
        return cls(
            category=OptimizationCategory(data.get("category", "batch_size")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            priority=Priority(data.get("priority", "medium")),
            estimated_speedup=str(data.get("estimated_speedup", "")),
            code_change=str(data.get("code_change", "")),
            current_value=str(data.get("current_value", "")),
            recommended_value=str(data.get("recommended_value", "")),
            confidence=float(data.get("confidence", 0.5)),  # type: ignore[arg-type]
        )


@dataclass
class OptimizationReport:
    """Full optimization advisory report."""

    optimizations: List[Optimization]
    total_estimated_speedup: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizations": [o.to_dict() for o in self.optimizations],
            "total_estimated_speedup": self.total_estimated_speedup,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizationReport:
        raw_opts = data.get("optimizations", [])
        optimizations = [
            Optimization.from_dict(o) for o in raw_opts
        ] if isinstance(raw_opts, list) else []
        return cls(
            optimizations=optimizations,
            total_estimated_speedup=str(data.get("total_estimated_speedup", "")),
            summary=str(data.get("summary", "")),
        )


# ============================================================================
# Thresholds
# ============================================================================

# GPU utilization below this suggests batch size can increase.
_LOW_UTIL_PCT: float = 60.0

# Memory headroom (MB) needed to safely increase batch size.
_MIN_HEADROOM_MB: float = 512.0

# Data loading idle time threshold (ms).
_DATA_LOADING_IDLE_MS: float = 5.0

# Communication overhead threshold.
_HIGH_COMM_OVERHEAD_PCT: float = 20.0

# Memory usage threshold for gradient checkpointing recommendation.
_HIGH_MEMORY_PCT: float = 75.0


# ============================================================================
# Optimization Advisor
# ============================================================================


class OptimizationAdvisor:
    """Generate specific optimization recommendations from profiling data.

    Usage::

        advisor = OptimizationAdvisor(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            data_result=data_result,
            distributed_result=dist_result,
            current_batch_size=32,
            current_num_workers=4,
        )
        report = advisor.analyze()
        for opt in report.optimizations:
            print(f"[{opt.priority}] {opt.title}")
            print(f"  Change: {opt.current_value} -> {opt.recommended_value}")
            print(f"  Code: {opt.code_change}")
    """

    def __init__(
        self,
        gpu_result: Optional[GPUProfileResult] = None,
        kernel_result: Optional[KernelTraceResult] = None,
        memory_result: Optional[MemoryAnalysisResult] = None,
        data_result: Optional[DataLoadingResult] = None,
        distributed_result: Optional[DistributedProfileResult] = None,
        current_batch_size: Optional[int] = None,
        current_num_workers: Optional[int] = None,
        current_precision: str = "fp32",
        num_gpus: int = 1,
        gpu_memory_gb: float = 80.0,
    ) -> None:
        self._gpu_result = gpu_result
        self._kernel_result = kernel_result
        self._memory_result = memory_result
        self._data_result = data_result
        self._distributed_result = distributed_result
        self._current_batch_size = current_batch_size
        self._current_num_workers = current_num_workers
        self._current_precision = current_precision
        self._num_gpus = max(num_gpus, 1)
        self._gpu_memory_gb = gpu_memory_gb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> OptimizationReport:
        """Run the optimization analysis and return recommendations."""
        optimizations: List[Optimization] = []

        optimizations.extend(self._advise_batch_size())
        optimizations.extend(self._advise_precision())
        optimizations.extend(self._advise_gradient_checkpointing())
        optimizations.extend(self._advise_data_loading())
        optimizations.extend(self._advise_compilation())
        optimizations.extend(self._advise_distributed())
        optimizations.extend(self._advise_memory())

        # Sort by priority.
        priority_order = {
            Priority.critical: 0,
            Priority.high: 1,
            Priority.medium: 2,
            Priority.low: 3,
        }
        optimizations.sort(key=lambda o: priority_order[o.priority])

        total_speedup = self._estimate_total_speedup(optimizations)
        summary = self._build_summary(optimizations, total_speedup)

        return OptimizationReport(
            optimizations=optimizations,
            total_estimated_speedup=total_speedup,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Batch size advisor
    # ------------------------------------------------------------------

    def _advise_batch_size(self) -> List[Optimization]:
        """Recommend batch size changes."""
        opts: List[Optimization] = []

        if self._gpu_result is None or not self._gpu_result.snapshots:
            return opts

        avg_util = self._gpu_result.avg_utilization
        peak_mem = self._gpu_result.peak_memory_mb
        total_mem = self._gpu_memory_gb * 1024.0  # GB -> MB

        if avg_util >= _LOW_UTIL_PCT:
            return opts

        # Estimate how much we can scale up.
        headroom_mb = total_mem - peak_mem
        if headroom_mb < _MIN_HEADROOM_MB:
            return opts

        # Heuristic: scale batch size by memory headroom ratio.
        if peak_mem > 0:
            scale_factor = min(total_mem * 0.85 / peak_mem, 4.0)
        else:
            scale_factor = 2.0

        current_bs = self._current_batch_size or 32
        recommended_bs = int(current_bs * scale_factor)
        # Round to nearest power of 2 or multiple of 8.
        recommended_bs = max(
            current_bs + 8,
            self._round_batch_size(recommended_bs),
        )

        opts.append(Optimization(
            category=OptimizationCategory.batch_size,
            title="Increase batch size",
            description=(
                f"GPU utilization is {avg_util:.0f}% with "
                f"{headroom_mb:.0f} MB memory headroom. Increasing batch "
                f"size will better saturate GPU compute and improve "
                f"throughput."
            ),
            priority=Priority.high if avg_util < 40.0 else Priority.medium,
            estimated_speedup=f"{min(scale_factor, 2.5):.1f}x",
            code_change=(
                f"# In your training config:\n"
                f"batch_size = {recommended_bs}  # was {current_bs}"
            ),
            current_value=str(current_bs),
            recommended_value=str(recommended_bs),
            confidence=0.8 if headroom_mb > 2000 else 0.6,
        ))

        return opts

    # ------------------------------------------------------------------
    # Precision advisor
    # ------------------------------------------------------------------

    def _advise_precision(self) -> List[Optimization]:
        """Recommend precision changes (BF16/FP16/TF32)."""
        opts: List[Optimization] = []

        if self._current_precision in ("bf16", "fp16"):
            return opts

        # Check if kernel data shows non-tensor-core kernels.
        has_kernel_data = (
            self._kernel_result is not None
            and self._kernel_result.total_kernel_time_us > 0
        )

        # Recommend BF16 for FP32 workloads.
        if self._current_precision == "fp32":
            opts.append(Optimization(
                category=OptimizationCategory.precision,
                title="Switch to BF16 mixed precision",
                description=(
                    "Training in FP32 uses 2x the memory bandwidth and "
                    "misses tensor core acceleration. BF16 maintains the "
                    "dynamic range of FP32 while halving memory usage."
                ),
                priority=Priority.high,
                estimated_speedup="1.5-2.0x",
                code_change=(
                    "# Add at the start of your training script:\n"
                    "from torch.cuda.amp import autocast, GradScaler\n"
                    "scaler = GradScaler()\n\n"
                    "# Wrap forward pass:\n"
                    "with autocast(dtype=torch.bfloat16):\n"
                    "    output = model(input)\n"
                    "    loss = criterion(output, target)\n\n"
                    "# Scale backward:\n"
                    "scaler.scale(loss).backward()\n"
                    "scaler.step(optimizer)\n"
                    "scaler.update()"
                ),
                current_value="fp32",
                recommended_value="bf16",
                confidence=0.9,
            ))

        # Always recommend TF32 if on Ampere+.
        if self._current_precision == "fp32":
            opts.append(Optimization(
                category=OptimizationCategory.precision,
                title="Enable TF32 for matmul operations",
                description=(
                    "TF32 provides up to 8x throughput improvement for "
                    "matrix multiplications on Ampere+ GPUs with minimal "
                    "precision loss."
                ),
                priority=Priority.medium,
                estimated_speedup="1.3-1.5x",
                code_change=(
                    "# Add at the start of your script (before any CUDA ops):\n"
                    "torch.backends.cuda.matmul.allow_tf32 = True\n"
                    "torch.backends.cudnn.allow_tf32 = True"
                ),
                current_value="tf32_disabled",
                recommended_value="tf32_enabled",
                confidence=0.85,
            ))

        return opts

    # ------------------------------------------------------------------
    # Gradient checkpointing advisor
    # ------------------------------------------------------------------

    def _advise_gradient_checkpointing(self) -> List[Optimization]:
        """Recommend gradient checkpointing when memory is tight."""
        opts: List[Optimization] = []

        if self._memory_result is None:
            if self._gpu_result is None:
                return opts
            peak_mem = self._gpu_result.peak_memory_mb
            total_mem = self._gpu_memory_gb * 1024.0
        else:
            # MemoryAnalysisResult stores peak in peak_snapshot or oom_risk.
            if self._memory_result.peak_snapshot is not None:
                peak_mem = self._memory_result.peak_snapshot.peak_allocated_mb
            elif self._memory_result.oom_risk.peak_usage_pct > 0:
                # Derive peak from usage percentage and configured total.
                total_mem = self._gpu_memory_gb * 1024.0
                peak_mem = total_mem * self._memory_result.oom_risk.peak_usage_pct / 100.0
            else:
                return opts
            total_mem = self._gpu_memory_gb * 1024.0

        if total_mem <= 0:
            return opts

        usage_pct = (peak_mem / total_mem) * 100.0
        if usage_pct < _HIGH_MEMORY_PCT:
            return opts

        opts.append(Optimization(
            category=OptimizationCategory.gradient_checkpointing,
            title="Enable gradient checkpointing",
            description=(
                f"GPU memory usage is {usage_pct:.0f}% "
                f"({peak_mem:.0f}/{total_mem:.0f} MB). Gradient "
                f"checkpointing trades ~30% compute for ~60% activation "
                f"memory savings, freeing memory for larger batch sizes."
            ),
            priority=Priority.high if usage_pct > 90 else Priority.medium,
            estimated_speedup=(
                "1.3-2.0x (via larger batch size after memory savings)"
            ),
            code_change=(
                "# For HuggingFace models:\n"
                "model.gradient_checkpointing_enable()\n\n"
                "# For custom PyTorch models:\n"
                "from torch.utils.checkpoint import checkpoint\n"
                "# In forward():\n"
                "output = checkpoint(self.layer, input, "
                "use_reentrant=False)"
            ),
            current_value="disabled",
            recommended_value="enabled",
            confidence=0.85,
        ))

        return opts

    # ------------------------------------------------------------------
    # Data loading advisor
    # ------------------------------------------------------------------

    def _advise_data_loading(self) -> List[Optimization]:
        """Recommend data loading optimizations."""
        opts: List[Optimization] = []

        if self._data_result is None:
            return opts

        idle_ms = self._data_result.avg_gpu_idle_time_ms
        current_workers = self._current_num_workers or 0

        if idle_ms < _DATA_LOADING_IDLE_MS and not self._data_result.is_bottleneck:
            return opts

        # Recommend optimal number of workers.
        import os
        cpu_count = os.cpu_count() or 8
        # Rule of thumb: num_workers = min(cpu_count, 4 * num_gpus)
        recommended_workers = min(cpu_count, 4 * self._num_gpus)
        recommended_workers = max(recommended_workers, 4)

        if current_workers < recommended_workers:
            opts.append(Optimization(
                category=OptimizationCategory.data_loading,
                title="Increase DataLoader workers",
                description=(
                    f"GPU idle time is {idle_ms:.1f} ms per batch waiting "
                    f"for data. Current workers: {current_workers}. "
                    f"More workers will pre-fetch data to keep the GPU fed."
                ),
                priority=(
                    Priority.high if idle_ms > 20 else Priority.medium
                ),
                estimated_speedup="1.2-1.5x",
                code_change=(
                    f"dataloader = DataLoader(\n"
                    f"    dataset,\n"
                    f"    batch_size=batch_size,\n"
                    f"    num_workers={recommended_workers},  "
                    f"# was {current_workers}\n"
                    f"    pin_memory=True,\n"
                    f"    persistent_workers=True,\n"
                    f"    prefetch_factor=2,\n"
                    f")"
                ),
                current_value=str(current_workers),
                recommended_value=str(recommended_workers),
                confidence=0.75,
            ))

        # Recommend pin_memory and prefetch if applicable.
        if idle_ms > _DATA_LOADING_IDLE_MS:
            opts.append(Optimization(
                category=OptimizationCategory.data_loading,
                title="Enable pin_memory and persistent_workers",
                description=(
                    "Pinned memory enables faster CPU-to-GPU transfers via "
                    "DMA. Persistent workers avoid fork overhead per epoch."
                ),
                priority=Priority.medium,
                estimated_speedup="1.1-1.2x",
                code_change=(
                    "dataloader = DataLoader(\n"
                    "    dataset,\n"
                    "    pin_memory=True,       # enable DMA transfers\n"
                    "    persistent_workers=True, # keep workers alive\n"
                    "    prefetch_factor=2,      # prefetch 2 batches\n"
                    ")"
                ),
                current_value="pin_memory=False",
                recommended_value="pin_memory=True, persistent_workers=True",
                confidence=0.8,
            ))

        return opts

    # ------------------------------------------------------------------
    # Compilation advisor
    # ------------------------------------------------------------------

    def _advise_compilation(self) -> List[Optimization]:
        """Recommend torch.compile and related optimizations."""
        opts: List[Optimization] = []

        if self._kernel_result is None:
            return opts

        # Check for high elementwise/reduction fraction.
        breakdown = self._kernel_result.category_breakdown
        total_us = self._kernel_result.total_kernel_time_us
        if total_us <= 0:
            return opts

        elementwise_us = (
            breakdown.get("elementwise", 0.0)
            + breakdown.get("reduction", 0.0)
        )
        elementwise_pct = (elementwise_us / total_us) * 100.0

        # Check for many inefficient kernels.
        inefficient = self._kernel_result.inefficient_kernels
        inefficient_pct = (
            sum(k.duration_us for k in inefficient) / total_us * 100.0
            if inefficient else 0.0
        )

        if elementwise_pct > 30.0 or inefficient_pct > 15.0:
            opts.append(Optimization(
                category=OptimizationCategory.compilation,
                title="Enable torch.compile()",
                description=(
                    f"Elementwise/reduction kernels consume "
                    f"{elementwise_pct:.0f}% of GPU time. torch.compile() "
                    f"fuses these into optimized kernels, reducing memory "
                    f"traffic and kernel launch overhead."
                ),
                priority=Priority.high if elementwise_pct > 50 else Priority.medium,
                estimated_speedup="1.2-1.5x",
                code_change=(
                    "# Add after model creation:\n"
                    "model = torch.compile(\n"
                    "    model,\n"
                    "    mode='max-autotune',  # best performance\n"
                    "    fullgraph=True,       # compile entire graph\n"
                    ")"
                ),
                current_value="uncompiled",
                recommended_value="torch.compile(mode='max-autotune')",
                confidence=0.7,
            ))

        return opts

    # ------------------------------------------------------------------
    # Distributed training advisor
    # ------------------------------------------------------------------

    def _advise_distributed(self) -> List[Optimization]:
        """Recommend distributed training optimizations."""
        opts: List[Optimization] = []

        if self._distributed_result is None:
            return opts

        overhead = self._distributed_result.gradient_sync_overhead_pct
        ratio = self._distributed_result.comm_compute_ratio
        world_size = self._distributed_result.world_size

        if overhead < _HIGH_COMM_OVERHEAD_PCT:
            return opts

        # Recommend gradient compression.
        if ratio > 0.3:
            opts.append(Optimization(
                category=OptimizationCategory.distributed,
                title="Enable gradient compression (PowerSGD)",
                description=(
                    f"Communication/compute ratio is {ratio:.2f} with "
                    f"{overhead:.0f}% sync overhead across "
                    f"{world_size} GPUs. PowerSGD compresses gradients "
                    f"to reduce communication volume."
                ),
                priority=Priority.high,
                estimated_speedup="1.3-1.5x",
                code_change=(
                    "from torch.distributed.algorithms.ddp_comm_hooks "
                    "import powerSGD_hook\n\n"
                    "state = powerSGD_hook.PowerSGDState(\n"
                    "    process_group=None,\n"
                    "    matrix_approximation_rank=1,\n"
                    "    start_powerSGD_iter=10,\n"
                    ")\n"
                    "model.register_comm_hook(state, "
                    "powerSGD_hook.powerSGD_hook)"
                ),
                current_value=f"comm_ratio={ratio:.2f}",
                recommended_value="PowerSGD compression",
                confidence=0.75,
            ))

        # Recommend FSDP for large world sizes.
        if world_size > 4 and overhead > 30:
            opts.append(Optimization(
                category=OptimizationCategory.distributed,
                title="Switch from DDP to FSDP",
                description=(
                    f"With {world_size} GPUs and {overhead:.0f}% sync "
                    f"overhead, FSDP shards parameters and optimizer states "
                    f"across GPUs, reducing both memory and communication."
                ),
                priority=Priority.medium,
                estimated_speedup="1.2-1.8x",
                code_change=(
                    "from torch.distributed.fsdp import (\n"
                    "    FullyShardedDataParallel as FSDP,\n"
                    "    MixedPrecision,\n"
                    ")\n\n"
                    "mp_policy = MixedPrecision(\n"
                    "    param_dtype=torch.bfloat16,\n"
                    "    reduce_dtype=torch.bfloat16,\n"
                    "    buffer_dtype=torch.bfloat16,\n"
                    ")\n\n"
                    "model = FSDP(\n"
                    "    model,\n"
                    "    mixed_precision=mp_policy,\n"
                    "    use_orig_params=True,\n"
                    ")"
                ),
                current_value="DDP",
                recommended_value="FSDP",
                confidence=0.65,
            ))

        return opts

    # ------------------------------------------------------------------
    # Memory advisor
    # ------------------------------------------------------------------

    def _advise_memory(self) -> List[Optimization]:
        """Recommend memory-specific optimizations."""
        opts: List[Optimization] = []

        if self._memory_result is None:
            return opts

        # Check for memory leak.
        if self._memory_result.leak_detected:
            opts.append(Optimization(
                category=OptimizationCategory.memory,
                title="Fix GPU memory leak",
                description=(
                    "A GPU memory leak has been detected. Memory is growing "
                    "monotonically which will eventually cause an OOM crash."
                ),
                priority=Priority.critical,
                estimated_speedup="prevents OOM crash",
                code_change=(
                    "# Common fixes:\n"
                    "# 1. Use .item() when logging losses:\n"
                    "log_loss = loss.item()  # not: log_loss = loss\n\n"
                    "# 2. Detach tensors before storing:\n"
                    "stored = tensor.detach()  # not: stored = tensor\n\n"
                    "# 3. Clear cache periodically:\n"
                    "if step % 100 == 0:\n"
                    "    torch.cuda.empty_cache()"
                ),
                current_value="memory_leak_detected",
                recommended_value="no_leak",
                confidence=0.9,
            ))

        # Check for high optimizer state memory.
        if self._memory_result.breakdown is not None:
            bd = self._memory_result.breakdown
            total = (
                bd.parameter_memory_mb
                + bd.gradient_memory_mb
                + bd.optimizer_state_mb
                + bd.activation_memory_mb
                + bd.other_mb
            )
            if total > 0 and bd.optimizer_state_mb > 2.0 * bd.parameter_memory_mb:
                opts.append(Optimization(
                    category=OptimizationCategory.memory,
                    title="Use 8-bit optimizer",
                    description=(
                        f"Optimizer states use {bd.optimizer_state_mb:.0f} MB "
                        f"(>2x parameter memory of "
                        f"{bd.parameter_memory_mb:.0f} MB). An 8-bit "
                        f"optimizer reduces this by 75%."
                    ),
                    priority=Priority.medium,
                    estimated_speedup="1.2-1.5x (via larger batch size)",
                    code_change=(
                        "# pip install bitsandbytes\n"
                        "import bitsandbytes as bnb\n\n"
                        "optimizer = bnb.optim.Adam8bit(\n"
                        "    model.parameters(),\n"
                        "    lr=1e-4,\n"
                        "    betas=(0.9, 0.999),\n"
                        ")"
                    ),
                    current_value=f"optimizer_state={bd.optimizer_state_mb:.0f}MB",
                    recommended_value="8-bit optimizer (~75% reduction)",
                    confidence=0.7,
                ))

        return opts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_batch_size(bs: int) -> int:
        """Round a batch size to a clean value (multiple of 8 or power of 2)."""
        # Find nearest power of 2.
        power_of_2 = 2 ** round(math.log2(max(bs, 1)))
        # Find nearest multiple of 8.
        multiple_of_8 = max(round(bs / 8) * 8, 8)
        # Return whichever is closer.
        if abs(power_of_2 - bs) < abs(multiple_of_8 - bs):
            return power_of_2
        return multiple_of_8

    @staticmethod
    def _estimate_total_speedup(optimizations: List[Optimization]) -> str:
        """Estimate combined speedup from all recommendations.

        Uses a conservative model: speedups are partially additive,
        not fully multiplicative.
        """
        if not optimizations:
            return "1.0x"

        # Extract lower bound of each speedup estimate.
        speedups: List[float] = []
        for opt in optimizations:
            s = opt.estimated_speedup
            # Parse "1.3-1.5x" -> 1.3
            try:
                if "-" in s:
                    lower = float(s.split("-")[0].strip().rstrip("x"))
                elif "x" in s:
                    lower = float(s.rstrip("x").strip())
                else:
                    continue
                if lower > 1.0:
                    speedups.append(lower)
            except (ValueError, IndexError):
                continue

        if not speedups:
            return "1.0x"

        # Conservative: largest speedup + 30% of remaining.
        speedups.sort(reverse=True)
        total = speedups[0]
        for s in speedups[1:]:
            total += (s - 1.0) * 0.3

        return f"{total:.1f}-{total * 1.3:.1f}x"

    @staticmethod
    def _build_summary(
        optimizations: List[Optimization],
        total_speedup: str,
    ) -> str:
        """Build a human-readable summary."""
        if not optimizations:
            return (
                "No significant optimizations identified. Your training "
                "configuration appears well-tuned."
            )

        critical = [o for o in optimizations if o.priority == Priority.critical]
        high = [o for o in optimizations if o.priority == Priority.high]

        parts: List[str] = [
            f"Found {len(optimizations)} optimization(s)."
        ]

        if critical:
            parts.append(
                f"{len(critical)} critical issue(s) require immediate attention."
            )

        if high:
            titles = [o.title for o in high[:3]]
            parts.append(f"Top recommendations: {', '.join(titles)}.")

        parts.append(f"Estimated combined speedup: {total_speedup}.")

        return " ".join(parts)
