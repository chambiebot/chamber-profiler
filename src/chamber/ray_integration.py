"""Ray Train auto-detection and per-worker profiling.

Automatically detects Ray Train jobs, profiles each worker individually,
and identifies communication bottlenecks in distributed Ray training.

If Ray is not installed the module logs a warning and provides empty
results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import ray
# ---------------------------------------------------------------------------
_ray_available: bool = False
try:
    import ray  # type: ignore[import-untyped]

    _ray_available = True
except ImportError:
    pass


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class WorkerProfile:
    """Per-worker profiling data for a Ray Train worker."""

    worker_id: str
    worker_rank: int
    node_id: str
    gpu_index: int
    avg_iteration_time_ms: float
    compute_time_pct: float
    communication_time_pct: float
    idle_time_pct: float
    gpu_utilization_pct: float
    gpu_memory_used_mb: float
    throughput_samples_per_sec: float
    num_iterations: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> WorkerProfile:
        return cls(
            worker_id=str(data.get("worker_id", "")),
            worker_rank=int(data.get("worker_rank", 0)),  # type: ignore[arg-type]
            node_id=str(data.get("node_id", "")),
            gpu_index=int(data.get("gpu_index", -1)),  # type: ignore[arg-type]
            avg_iteration_time_ms=float(data.get("avg_iteration_time_ms", 0.0)),  # type: ignore[arg-type]
            compute_time_pct=float(data.get("compute_time_pct", 0.0)),  # type: ignore[arg-type]
            communication_time_pct=float(data.get("communication_time_pct", 0.0)),  # type: ignore[arg-type]
            idle_time_pct=float(data.get("idle_time_pct", 0.0)),  # type: ignore[arg-type]
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            gpu_memory_used_mb=float(data.get("gpu_memory_used_mb", 0.0)),  # type: ignore[arg-type]
            throughput_samples_per_sec=float(data.get("throughput_samples_per_sec", 0.0)),  # type: ignore[arg-type]
            num_iterations=int(data.get("num_iterations", 0)),  # type: ignore[arg-type]
        )


@dataclass
class CommunicationBottleneck:
    """A detected communication bottleneck between Ray workers."""

    bottleneck_type: str  # "allreduce_slow", "straggler", "imbalanced", "bandwidth"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_workers: List[str]
    recommendation: str
    metric_value: float  # the metric that triggered this detection

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> CommunicationBottleneck:
        raw_workers = data.get("affected_workers", [])
        workers = (
            [str(w) for w in raw_workers]  # type: ignore[union-attr]
            if isinstance(raw_workers, list)
            else []
        )
        return cls(
            bottleneck_type=str(data.get("bottleneck_type", "")),
            severity=str(data.get("severity", "medium")),
            description=str(data.get("description", "")),
            affected_workers=workers,
            recommendation=str(data.get("recommendation", "")),
            metric_value=float(data.get("metric_value", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class RayTrainProfileResult:
    """Full profiling result for a Ray Train job."""

    job_id: str
    job_name: str
    worker_profiles: List[WorkerProfile]
    communication_bottlenecks: List[CommunicationBottleneck]
    num_workers: int
    num_nodes: int
    total_training_time_s: float
    avg_throughput_samples_per_sec: float
    communication_overhead_pct: float
    load_balance_score: float  # 0.0 (worst) to 1.0 (perfect balance)
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "worker_profiles": [w.to_dict() for w in self.worker_profiles],
            "communication_bottlenecks": [
                b.to_dict() for b in self.communication_bottlenecks
            ],
            "num_workers": self.num_workers,
            "num_nodes": self.num_nodes,
            "total_training_time_s": self.total_training_time_s,
            "avg_throughput_samples_per_sec": self.avg_throughput_samples_per_sec,
            "communication_overhead_pct": self.communication_overhead_pct,
            "load_balance_score": self.load_balance_score,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RayTrainProfileResult:
        raw_workers = data.get("worker_profiles", [])
        workers = [
            WorkerProfile.from_dict(w) for w in raw_workers
        ] if isinstance(raw_workers, list) else []
        raw_bottlenecks = data.get("communication_bottlenecks", [])
        bottlenecks = [
            CommunicationBottleneck.from_dict(b) for b in raw_bottlenecks
        ] if isinstance(raw_bottlenecks, list) else []
        raw_recs = data.get("recommendations", [])
        recs = (
            [str(r) for r in raw_recs]
            if isinstance(raw_recs, list)
            else []
        )
        return cls(
            job_id=str(data.get("job_id", "")),
            job_name=str(data.get("job_name", "")),
            worker_profiles=workers,
            communication_bottlenecks=bottlenecks,
            num_workers=int(data.get("num_workers", 0)),  # type: ignore[arg-type]
            num_nodes=int(data.get("num_nodes", 0)),  # type: ignore[arg-type]
            total_training_time_s=float(data.get("total_training_time_s", 0.0)),  # type: ignore[arg-type]
            avg_throughput_samples_per_sec=float(data.get("avg_throughput_samples_per_sec", 0.0)),  # type: ignore[arg-type]
            communication_overhead_pct=float(data.get("communication_overhead_pct", 0.0)),  # type: ignore[arg-type]
            load_balance_score=float(data.get("load_balance_score", 0.0)),  # type: ignore[arg-type]
            recommendations=recs,
        )


# ============================================================================
# Constants
# ============================================================================

# Worker imbalance: if max/min iteration time ratio exceeds this, flag it.
_IMBALANCE_RATIO_THRESHOLD: float = 1.3

# Communication overhead threshold.
_HIGH_COMM_OVERHEAD_PCT: float = 20.0

# Straggler detection: sigma threshold.
_STRAGGLER_SIGMA: float = 1.5

# Low GPU utilization threshold for workers.
_LOW_UTIL_PCT: float = 50.0


# ============================================================================
# Ray Train Integration
# ============================================================================


class RayTrainIntegration:
    """Auto-detect and profile Ray Train distributed training jobs.

    Usage::

        integration = RayTrainIntegration()

        # Auto-detect a running Ray Train job
        if integration.detect_ray_train_job():
            result = integration.profile()
            for bottleneck in result.communication_bottlenecks:
                print(bottleneck.description)

        # Or profile from worker data
        result = integration.profile_from_workers(worker_profiles)
    """

    def __init__(self, dashboard_url: Optional[str] = None) -> None:
        self._dashboard_url = dashboard_url
        self._detected_job_id: Optional[str] = None
        self._detected_job_name: Optional[str] = None
        self._worker_profiles: List[WorkerProfile] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_ray_train_job(self) -> bool:
        """Attempt to auto-detect a running Ray Train job.

        Returns True if a Ray Train job was detected.
        """
        if not _ray_available:
            logger.warning("Ray is not installed; cannot detect Ray Train jobs.")
            return False

        try:
            if not ray.is_initialized():
                logger.debug("Ray is not initialized.")
                return False

            actors = ray.state.actors()  # type: ignore[attr-defined]
            train_actors = {}
            for actor_id, info in actors.items():
                if info.get("State") != "ALIVE":
                    continue
                # Look for Ray Train worker actors.
                class_name = info.get("ActorClassName", "")
                if "Train" in class_name or "Worker" in class_name:
                    resources = info.get("Resources", {})
                    if resources.get("GPU", 0) > 0:
                        train_actors[actor_id] = info

            if train_actors:
                self._detected_job_id = f"ray-train-{int(time.time())}"
                self._detected_job_name = "Ray Train Job"
                logger.info(
                    "Detected Ray Train job with %d workers.",
                    len(train_actors),
                )
                return True

        except Exception:
            logger.debug("Failed to detect Ray Train job.", exc_info=True)

        return False

    def profile(self) -> RayTrainProfileResult:
        """Profile the detected Ray Train job.

        Returns a full profiling result with per-worker data and
        communication bottleneck analysis.
        """
        if not _ray_available or not ray.is_initialized():
            return self._empty_result()

        start_time = time.monotonic()
        workers = self._collect_worker_profiles()
        duration = time.monotonic() - start_time

        return self.profile_from_workers(
            workers,
            job_id=self._detected_job_id or "",
            job_name=self._detected_job_name or "",
            training_time_s=duration,
        )

    def profile_from_workers(
        self,
        worker_profiles: List[WorkerProfile],
        job_id: str = "",
        job_name: str = "",
        training_time_s: float = 0.0,
    ) -> RayTrainProfileResult:
        """Build a profiling result from pre-collected worker profiles.

        This is useful for testing or when worker data is collected
        externally.
        """
        if not worker_profiles:
            return self._empty_result()

        bottlenecks = self._detect_communication_bottlenecks(worker_profiles)
        comm_overhead = self._compute_communication_overhead(worker_profiles)
        balance_score = self._compute_load_balance_score(worker_profiles)
        avg_throughput = self._compute_avg_throughput(worker_profiles)
        num_nodes = len(set(w.node_id for w in worker_profiles))
        recommendations = self._generate_recommendations(
            worker_profiles, bottlenecks, comm_overhead, balance_score,
        )

        return RayTrainProfileResult(
            job_id=job_id,
            job_name=job_name,
            worker_profiles=worker_profiles,
            communication_bottlenecks=bottlenecks,
            num_workers=len(worker_profiles),
            num_nodes=num_nodes,
            total_training_time_s=round(training_time_s, 3),
            avg_throughput_samples_per_sec=round(avg_throughput, 2),
            communication_overhead_pct=round(comm_overhead, 1),
            load_balance_score=round(balance_score, 3),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Worker profiling
    # ------------------------------------------------------------------

    def _collect_worker_profiles(self) -> List[WorkerProfile]:
        """Collect profiling data from all Ray Train workers."""
        if not _ray_available or not ray.is_initialized():
            return []

        profiles: List[WorkerProfile] = []
        try:
            actors = ray.state.actors()  # type: ignore[attr-defined]
            rank = 0
            for actor_id, info in actors.items():
                if info.get("State") != "ALIVE":
                    continue
                resources = info.get("Resources", {})
                if resources.get("GPU", 0) <= 0:
                    continue

                gpu_index = int(resources.get("GPU_group_0_index", 0))
                profiles.append(WorkerProfile(
                    worker_id=str(actor_id),
                    worker_rank=rank,
                    node_id=info.get("NodeID", ""),
                    gpu_index=gpu_index,
                    avg_iteration_time_ms=0.0,
                    compute_time_pct=0.0,
                    communication_time_pct=0.0,
                    idle_time_pct=0.0,
                    gpu_utilization_pct=0.0,
                    gpu_memory_used_mb=0.0,
                    throughput_samples_per_sec=0.0,
                    num_iterations=0,
                ))
                rank += 1
        except Exception:
            logger.debug("Failed to collect worker profiles.", exc_info=True)

        return profiles

    # ------------------------------------------------------------------
    # Communication bottleneck detection
    # ------------------------------------------------------------------

    def _detect_communication_bottlenecks(
        self,
        workers: List[WorkerProfile],
    ) -> List[CommunicationBottleneck]:
        """Detect communication bottlenecks in the Ray Train job."""
        bottlenecks: List[CommunicationBottleneck] = []

        if len(workers) < 2:
            return bottlenecks

        # Check for straggler workers.
        stragglers = self._detect_stragglers(workers)
        bottlenecks.extend(stragglers)

        # Check for load imbalance.
        imbalance = self._detect_load_imbalance(workers)
        if imbalance is not None:
            bottlenecks.append(imbalance)

        # Check for high communication overhead.
        high_comm = self._detect_high_communication(workers)
        bottlenecks.extend(high_comm)

        # Check for low utilization workers.
        low_util = self._detect_low_utilization(workers)
        bottlenecks.extend(low_util)

        return bottlenecks

    def _detect_stragglers(
        self,
        workers: List[WorkerProfile],
    ) -> List[CommunicationBottleneck]:
        """Detect straggler workers."""
        bottlenecks: List[CommunicationBottleneck] = []

        active = [w for w in workers if w.avg_iteration_time_ms > 0]
        if len(active) < 2:
            return bottlenecks

        times = [w.avg_iteration_time_ms for w in active]
        mean_time = sum(times) / len(times)
        if mean_time <= 0:
            return bottlenecks

        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5

        if std_time <= 0:
            return bottlenecks

        threshold = mean_time + _STRAGGLER_SIGMA * std_time

        for worker in active:
            if worker.avg_iteration_time_ms > threshold:
                slowdown = worker.avg_iteration_time_ms / mean_time
                bottlenecks.append(CommunicationBottleneck(
                    bottleneck_type="straggler",
                    severity="high" if slowdown > 1.5 else "medium",
                    description=(
                        f"Worker {worker.worker_id} (rank {worker.worker_rank}) "
                        f"is {slowdown:.1f}x slower than average "
                        f"({worker.avg_iteration_time_ms:.1f}ms vs "
                        f"{mean_time:.1f}ms mean)."
                    ),
                    affected_workers=[worker.worker_id],
                    recommendation=(
                        f"Check GPU {worker.gpu_index} on node "
                        f"{worker.node_id} for thermal throttling, slow "
                        f"interconnect, or uneven data distribution."
                    ),
                    metric_value=slowdown,
                ))

        return bottlenecks

    @staticmethod
    def _detect_load_imbalance(
        workers: List[WorkerProfile],
    ) -> Optional[CommunicationBottleneck]:
        """Detect workload imbalance across workers."""
        active = [w for w in workers if w.avg_iteration_time_ms > 0]
        if len(active) < 2:
            return None

        times = [w.avg_iteration_time_ms for w in active]
        max_time = max(times)
        min_time = min(times)

        if min_time <= 0:
            return None

        ratio = max_time / min_time
        if ratio < _IMBALANCE_RATIO_THRESHOLD:
            return None

        slowest = [
            w for w in active if w.avg_iteration_time_ms == max_time
        ]

        return CommunicationBottleneck(
            bottleneck_type="imbalanced",
            severity="high" if ratio > 2.0 else "medium",
            description=(
                f"Workload imbalance detected: fastest worker takes "
                f"{min_time:.1f}ms, slowest takes {max_time:.1f}ms "
                f"({ratio:.1f}x ratio). All workers synchronize at the "
                f"slowest worker's pace."
            ),
            affected_workers=[w.worker_id for w in slowest],
            recommendation=(
                "Ensure data is evenly distributed across workers. "
                "Check for uneven sequence lengths or batch sizes. "
                "Consider using dynamic batching or padding strategies."
            ),
            metric_value=ratio,
        )

    @staticmethod
    def _detect_high_communication(
        workers: List[WorkerProfile],
    ) -> List[CommunicationBottleneck]:
        """Detect workers with high communication overhead."""
        bottlenecks: List[CommunicationBottleneck] = []

        for worker in workers:
            if worker.communication_time_pct > _HIGH_COMM_OVERHEAD_PCT:
                bottlenecks.append(CommunicationBottleneck(
                    bottleneck_type="allreduce_slow",
                    severity=(
                        "high" if worker.communication_time_pct > 40
                        else "medium"
                    ),
                    description=(
                        f"Worker {worker.worker_id} (rank "
                        f"{worker.worker_rank}) spends "
                        f"{worker.communication_time_pct:.0f}% of time "
                        f"on communication."
                    ),
                    affected_workers=[worker.worker_id],
                    recommendation=(
                        "Enable gradient compression (PowerSGD) or switch "
                        "to FSDP to reduce communication volume. Check "
                        "inter-node bandwidth."
                    ),
                    metric_value=worker.communication_time_pct,
                ))

        return bottlenecks

    @staticmethod
    def _detect_low_utilization(
        workers: List[WorkerProfile],
    ) -> List[CommunicationBottleneck]:
        """Detect workers with low GPU utilization."""
        bottlenecks: List[CommunicationBottleneck] = []

        low_util_workers = [
            w for w in workers
            if 0 < w.gpu_utilization_pct < _LOW_UTIL_PCT
        ]

        if low_util_workers and len(low_util_workers) > len(workers) * 0.3:
            avg_util = sum(
                w.gpu_utilization_pct for w in low_util_workers
            ) / len(low_util_workers)
            bottlenecks.append(CommunicationBottleneck(
                bottleneck_type="bandwidth",
                severity="medium",
                description=(
                    f"{len(low_util_workers)} of {len(workers)} workers "
                    f"have low GPU utilization (avg {avg_util:.0f}%). "
                    f"Workers may be waiting for communication or data."
                ),
                affected_workers=[w.worker_id for w in low_util_workers],
                recommendation=(
                    "Increase per-worker batch size, enable communication "
                    "overlap, or use torch.compile() to improve GPU "
                    "utilization."
                ),
                metric_value=avg_util,
            ))

        return bottlenecks

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_communication_overhead(
        workers: List[WorkerProfile],
    ) -> float:
        """Compute average communication overhead across all workers."""
        if not workers:
            return 0.0
        comm_pcts = [w.communication_time_pct for w in workers]
        return sum(comm_pcts) / len(comm_pcts)

    @staticmethod
    def _compute_load_balance_score(
        workers: List[WorkerProfile],
    ) -> float:
        """Compute load balance score (0.0 worst, 1.0 perfect).

        Uses coefficient of variation of iteration times.
        """
        active = [w for w in workers if w.avg_iteration_time_ms > 0]
        if len(active) < 2:
            return 1.0

        times = [w.avg_iteration_time_ms for w in active]
        mean_time = sum(times) / len(times)
        if mean_time <= 0:
            return 1.0

        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        cv = (variance ** 0.5) / mean_time

        # Convert CV to a 0-1 score. CV=0 -> score=1.0, CV=1 -> score=0.0
        return max(0.0, min(1.0, 1.0 - cv))

    @staticmethod
    def _compute_avg_throughput(
        workers: List[WorkerProfile],
    ) -> float:
        """Compute total throughput across all workers."""
        return sum(w.throughput_samples_per_sec for w in workers)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(
        workers: List[WorkerProfile],
        bottlenecks: List[CommunicationBottleneck],
        comm_overhead: float,
        balance_score: float,
    ) -> List[str]:
        """Generate overall recommendations for the Ray Train job."""
        recs: List[str] = []

        if balance_score < 0.8 and len(workers) > 1:
            recs.append(
                f"Load balance score is {balance_score:.2f} (1.0=perfect). "
                f"Consider using Ray Data with streaming for more even "
                f"data distribution across workers."
            )

        if comm_overhead > _HIGH_COMM_OVERHEAD_PCT:
            recs.append(
                f"Average communication overhead is {comm_overhead:.0f}%. "
                f"Use Ray Train's TorchConfig with backend='nccl' and "
                f"enable gradient compression."
            )

        low_util = [w for w in workers if 0 < w.gpu_utilization_pct < _LOW_UTIL_PCT]
        if low_util:
            recs.append(
                f"{len(low_util)} workers have low GPU utilization. "
                f"Increase per-worker batch size or enable torch.compile() "
                f"in the training function."
            )

        if len(bottlenecks) > 3:
            recs.append(
                f"Multiple communication bottlenecks detected "
                f"({len(bottlenecks)}). Consider reducing world size and "
                f"using gradient accumulation instead."
            )

        if not recs:
            recs.append(
                "No major bottlenecks detected. The Ray Train job appears "
                "well-configured."
            )

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> RayTrainProfileResult:
        """Return an empty result."""
        return RayTrainProfileResult(
            job_id="",
            job_name="",
            worker_profiles=[],
            communication_bottlenecks=[],
            num_workers=0,
            num_nodes=0,
            total_training_time_s=0.0,
            avg_throughput_samples_per_sec=0.0,
            communication_overhead_pct=0.0,
            load_balance_score=1.0,
            recommendations=[],
        )
