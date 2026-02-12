"""Multi-node distributed training profiler.

Profiles NCCL allreduce time, gradient sync overhead, and the
communication vs compute ratio for multi-node distributed training.

Works by recording timing around gradient synchronization points and
computing aggregate statistics. If PyTorch distributed is not available,
the profiler produces empty results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch.distributed
# ---------------------------------------------------------------------------
_torch_available: bool = False
_dist_available: bool = False
try:
    import torch  # type: ignore[import-untyped]

    _torch_available = True
except ImportError:
    pass

if _torch_available:
    try:
        import torch.distributed as dist  # type: ignore[import-untyped]

        _dist_available = True
    except ImportError:
        pass


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class AllReduceEvent:
    """A single recorded allreduce operation with timing."""

    timestamp: float
    duration_us: float
    data_size_bytes: int
    step_index: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> AllReduceEvent:
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
            duration_us=float(data.get("duration_us", 0.0)),  # type: ignore[arg-type]
            data_size_bytes=int(data.get("data_size_bytes", 0)),  # type: ignore[arg-type]
            step_index=int(data.get("step_index", 0)),  # type: ignore[arg-type]
        )


@dataclass
class GradSyncEvent:
    """Timing for a single gradient synchronization step."""

    step_index: int
    compute_time_us: float
    sync_time_us: float
    total_step_time_us: float
    timestamp: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> GradSyncEvent:
        return cls(
            step_index=int(data.get("step_index", 0)),  # type: ignore[arg-type]
            compute_time_us=float(data.get("compute_time_us", 0.0)),  # type: ignore[arg-type]
            sync_time_us=float(data.get("sync_time_us", 0.0)),  # type: ignore[arg-type]
            total_step_time_us=float(data.get("total_step_time_us", 0.0)),  # type: ignore[arg-type]
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class DistributedProfileResult:
    """Aggregated output of a distributed training profiling session."""

    allreduce_events: List[AllReduceEvent]
    grad_sync_events: List[GradSyncEvent]
    total_allreduce_time_us: float
    avg_allreduce_time_us: float
    total_grad_sync_time_us: float
    avg_grad_sync_time_us: float
    total_compute_time_us: float
    comm_compute_ratio: float
    gradient_sync_overhead_pct: float
    world_size: int
    num_nodes: int
    recommendations: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "allreduce_events": [e.to_dict() for e in self.allreduce_events],
            "grad_sync_events": [e.to_dict() for e in self.grad_sync_events],
            "total_allreduce_time_us": self.total_allreduce_time_us,
            "avg_allreduce_time_us": self.avg_allreduce_time_us,
            "total_grad_sync_time_us": self.total_grad_sync_time_us,
            "avg_grad_sync_time_us": self.avg_grad_sync_time_us,
            "total_compute_time_us": self.total_compute_time_us,
            "comm_compute_ratio": self.comm_compute_ratio,
            "gradient_sync_overhead_pct": self.gradient_sync_overhead_pct,
            "world_size": self.world_size,
            "num_nodes": self.num_nodes,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> DistributedProfileResult:
        raw_ar = data.get("allreduce_events", [])
        allreduce_events = [
            AllReduceEvent.from_dict(e)  # type: ignore[arg-type]
            for e in raw_ar  # type: ignore[union-attr]
        ]
        raw_gs = data.get("grad_sync_events", [])
        grad_sync_events = [
            GradSyncEvent.from_dict(e)  # type: ignore[arg-type]
            for e in raw_gs  # type: ignore[union-attr]
        ]
        raw_recs = data.get("recommendations", [])
        recommendations: List[str] = (
            [str(r) for r in raw_recs]  # type: ignore[union-attr]
            if isinstance(raw_recs, list)
            else []
        )
        return cls(
            allreduce_events=allreduce_events,
            grad_sync_events=grad_sync_events,
            total_allreduce_time_us=float(data.get("total_allreduce_time_us", 0.0)),  # type: ignore[arg-type]
            avg_allreduce_time_us=float(data.get("avg_allreduce_time_us", 0.0)),  # type: ignore[arg-type]
            total_grad_sync_time_us=float(data.get("total_grad_sync_time_us", 0.0)),  # type: ignore[arg-type]
            avg_grad_sync_time_us=float(data.get("avg_grad_sync_time_us", 0.0)),  # type: ignore[arg-type]
            total_compute_time_us=float(data.get("total_compute_time_us", 0.0)),  # type: ignore[arg-type]
            comm_compute_ratio=float(data.get("comm_compute_ratio", 0.0)),  # type: ignore[arg-type]
            gradient_sync_overhead_pct=float(data.get("gradient_sync_overhead_pct", 0.0)),  # type: ignore[arg-type]
            world_size=int(data.get("world_size", 0)),  # type: ignore[arg-type]
            num_nodes=int(data.get("num_nodes", 0)),  # type: ignore[arg-type]
            recommendations=recommendations,
        )


# ============================================================================
# Distributed Profiler
# ============================================================================


class DistributedProfiler:
    """Profile multi-node distributed training.

    Records NCCL allreduce timing, gradient synchronization overhead,
    and communication vs compute ratio.

    Usage::

        profiler = DistributedProfiler()
        profiler.start()
        for step in range(num_steps):
            loss = model(batch)
            loss.backward()
            profiler.record_step(step, compute_start, compute_end)
            optimizer.step()
        result = profiler.stop()
        print(result.comm_compute_ratio, result.gradient_sync_overhead_pct)
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._start_time: float = 0.0
        self._allreduce_events: List[AllReduceEvent] = []
        self._grad_sync_events: List[GradSyncEvent] = []
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin profiling distributed training."""
        if self._active:
            logger.warning(
                "DistributedProfiler is already active; ignoring duplicate start()."
            )
            return

        self._allreduce_events = []
        self._grad_sync_events = []
        self._step_counter = 0
        self._start_time = time.monotonic()

        if not _dist_available:
            logger.warning(
                "torch.distributed is not available. Distributed profiling "
                "is disabled; stop() will return empty results."
            )

        self._active = True
        logger.debug("DistributedProfiler started.")

    def stop(self) -> DistributedProfileResult:
        """Stop profiling and return the result."""
        if not self._active:
            logger.warning(
                "DistributedProfiler.stop() called but profiler is not active."
            )
            return self._empty_result()

        self._active = False

        allreduce_events = list(self._allreduce_events)
        grad_sync_events = list(self._grad_sync_events)

        total_ar_time = sum(e.duration_us for e in allreduce_events)
        avg_ar_time = total_ar_time / len(allreduce_events) if allreduce_events else 0.0

        total_sync_time = sum(e.sync_time_us for e in grad_sync_events)
        avg_sync_time = total_sync_time / len(grad_sync_events) if grad_sync_events else 0.0
        total_compute = sum(e.compute_time_us for e in grad_sync_events)

        total_step_time = sum(e.total_step_time_us for e in grad_sync_events)
        comm_compute_ratio = total_sync_time / total_compute if total_compute > 0 else 0.0
        overhead_pct = (total_sync_time / total_step_time * 100.0) if total_step_time > 0 else 0.0

        world_size = self._get_world_size()
        num_nodes = self._get_num_nodes()

        recommendations = self._generate_recommendations(
            comm_compute_ratio, overhead_pct, allreduce_events, world_size,
        )

        return DistributedProfileResult(
            allreduce_events=allreduce_events,
            grad_sync_events=grad_sync_events,
            total_allreduce_time_us=round(total_ar_time, 2),
            avg_allreduce_time_us=round(avg_ar_time, 2),
            total_grad_sync_time_us=round(total_sync_time, 2),
            avg_grad_sync_time_us=round(avg_sync_time, 2),
            total_compute_time_us=round(total_compute, 2),
            comm_compute_ratio=round(comm_compute_ratio, 4),
            gradient_sync_overhead_pct=round(overhead_pct, 2),
            world_size=world_size,
            num_nodes=num_nodes,
            recommendations=recommendations,
        )

    def record_allreduce(
        self,
        duration_us: float,
        data_size_bytes: int = 0,
        step_index: Optional[int] = None,
    ) -> None:
        """Record an allreduce operation.

        Parameters
        ----------
        duration_us:
            Wall-clock duration of the allreduce in microseconds.
        data_size_bytes:
            Size of data transferred in the allreduce.
        step_index:
            Training step index. Auto-incremented if not provided.
        """
        if not self._active:
            return

        idx = step_index if step_index is not None else self._step_counter
        self._allreduce_events.append(AllReduceEvent(
            timestamp=time.time(),
            duration_us=duration_us,
            data_size_bytes=data_size_bytes,
            step_index=idx,
        ))

    def record_step(
        self,
        step_index: int,
        compute_time_us: float,
        sync_time_us: float,
    ) -> None:
        """Record timing for a single training step.

        Parameters
        ----------
        step_index:
            Training step number.
        compute_time_us:
            Time spent on forward + backward computation in microseconds.
        sync_time_us:
            Time spent on gradient synchronization in microseconds.
        """
        if not self._active:
            return

        total = compute_time_us + sync_time_us
        self._grad_sync_events.append(GradSyncEvent(
            step_index=step_index,
            compute_time_us=compute_time_us,
            sync_time_us=sync_time_us,
            total_step_time_us=total,
            timestamp=time.time(),
        ))
        self._step_counter = step_index + 1

    def get_comm_compute_ratio(self) -> float:
        """Calculate the current communication to compute ratio.

        Returns
        -------
        float
            Ratio of total sync time to total compute time. A value > 1.0
            means more time is spent communicating than computing.
        """
        total_sync = sum(e.sync_time_us for e in self._grad_sync_events)
        total_compute = sum(e.compute_time_us for e in self._grad_sync_events)
        if total_compute <= 0:
            return 0.0
        return total_sync / total_compute

    def get_gradient_sync_overhead(self) -> float:
        """Calculate gradient sync overhead as a percentage of total step time.

        Returns
        -------
        float
            Percentage of total step time spent on gradient synchronization.
        """
        total_sync = sum(e.sync_time_us for e in self._grad_sync_events)
        total_step = sum(e.total_step_time_us for e in self._grad_sync_events)
        if total_step <= 0:
            return 0.0
        return (total_sync / total_step) * 100.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_world_size() -> int:
        """Return the distributed world size."""
        if not _dist_available:
            return 0
        try:
            if dist.is_initialized():
                return dist.get_world_size()
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_num_nodes() -> int:
        """Estimate the number of nodes in the distributed group.

        Uses the local world size to infer node count:
        num_nodes = world_size / local_world_size.
        """
        if not _dist_available:
            return 0
        try:
            if not dist.is_initialized():
                return 0
            world_size = dist.get_world_size()
            # Try to get local world size from environment or process group.
            import os
            local_size_str = os.environ.get("LOCAL_WORLD_SIZE", "")
            if local_size_str:
                local_size = int(local_size_str)
                if local_size > 0:
                    return max(world_size // local_size, 1)
            return 1
        except Exception:
            pass
        return 0

    @staticmethod
    def _generate_recommendations(
        comm_compute_ratio: float,
        overhead_pct: float,
        allreduce_events: List[AllReduceEvent],
        world_size: int,
    ) -> List[str]:
        """Generate recommendations based on distributed profiling data."""
        recommendations: List[str] = []

        if comm_compute_ratio > 0.5:
            recommendations.append(
                f"Communication/compute ratio is {comm_compute_ratio:.2f}. "
                f"Consider increasing batch size per GPU to improve the ratio, "
                f"or use gradient compression (e.g. PowerSGD)."
            )

        if overhead_pct > 30.0:
            recommendations.append(
                f"Gradient sync overhead is {overhead_pct:.1f}% of step time. "
                f"Enable gradient bucketing (bucket_cap_mb=50+) and "
                f"gradient_as_bucket_view=True in DDP."
            )

        if overhead_pct > 50.0:
            recommendations.append(
                "Consider switching from DDP to FSDP to shard parameters and "
                "reduce per-GPU communication volume."
            )

        if allreduce_events:
            sizes = [e.data_size_bytes for e in allreduce_events if e.data_size_bytes > 0]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                if avg_size < 1_000_000:  # < 1MB
                    recommendations.append(
                        f"Average allreduce message size is small "
                        f"({avg_size / 1024:.1f} KB). Increase gradient bucket "
                        f"size to reduce latency overhead."
                    )

        if world_size > 8 and overhead_pct > 20.0:
            recommendations.append(
                f"With {world_size} GPUs and {overhead_pct:.1f}% sync overhead, "
                f"consider hierarchical allreduce (NCCL_ALGO=Tree) or "
                f"reducing world size with gradient accumulation."
            )

        return recommendations

    @staticmethod
    def _empty_result() -> DistributedProfileResult:
        """Return an empty result."""
        return DistributedProfileResult(
            allreduce_events=[],
            grad_sync_events=[],
            total_allreduce_time_us=0.0,
            avg_allreduce_time_us=0.0,
            total_grad_sync_time_us=0.0,
            avg_grad_sync_time_us=0.0,
            total_compute_time_us=0.0,
            comm_compute_ratio=0.0,
            gradient_sync_overhead_pct=0.0,
            world_size=0,
            num_nodes=0,
            recommendations=[],
        )
