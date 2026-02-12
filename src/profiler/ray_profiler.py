"""Ray Train job profiling module.

Profiles Ray Train distributed training jobs by scraping Ray metrics,
mapping actors to GPUs, and detecting stragglers in distributed Ray
training runs.

If Ray is not installed the profiler logs a warning and produces empty
results -- the rest of the profiling suite keeps working.
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
    logger.debug("ray is available; Ray Train profiling is supported.")
except ImportError:
    logger.debug("ray is not installed; Ray Train profiling will produce empty results.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Ray dashboard metrics endpoint (relative to dashboard URL).
_DEFAULT_METRICS_PORT: int = 8265

# Straggler detection: an actor whose iteration time exceeds
# mean + threshold * std is a straggler.
_STRAGGLER_SIGMA_THRESHOLD: float = 1.5

# Minimum number of actors needed for straggler detection.
_MIN_ACTORS_FOR_STRAGGLER: int = 2


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class RayActorMetrics:
    """Metrics for a single Ray Train actor (worker)."""

    actor_id: str
    node_id: str
    gpu_index: int
    pid: int
    avg_iteration_time_ms: float
    total_iterations: int
    gpu_utilization_pct: float
    gpu_memory_used_mb: float
    status: str  # "running", "idle", "finished", "error"
    timestamp: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RayActorMetrics:
        return cls(
            actor_id=str(data.get("actor_id", "")),
            node_id=str(data.get("node_id", "")),
            gpu_index=int(data.get("gpu_index", -1)),  # type: ignore[arg-type]
            pid=int(data.get("pid", 0)),  # type: ignore[arg-type]
            avg_iteration_time_ms=float(data.get("avg_iteration_time_ms", 0.0)),  # type: ignore[arg-type]
            total_iterations=int(data.get("total_iterations", 0)),  # type: ignore[arg-type]
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0.0)),  # type: ignore[arg-type]
            gpu_memory_used_mb=float(data.get("gpu_memory_used_mb", 0.0)),  # type: ignore[arg-type]
            status=str(data.get("status", "unknown")),
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class RayActorGPUMapping:
    """Maps a Ray actor to a specific GPU on a specific node."""

    actor_id: str
    node_ip: str
    node_id: str
    gpu_index: int
    gpu_name: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RayActorGPUMapping:
        return cls(
            actor_id=str(data.get("actor_id", "")),
            node_ip=str(data.get("node_ip", "")),
            node_id=str(data.get("node_id", "")),
            gpu_index=int(data.get("gpu_index", -1)),  # type: ignore[arg-type]
            gpu_name=str(data.get("gpu_name", "")),
        )


@dataclass
class RayStragglerInfo:
    """Information about a detected straggler actor."""

    actor_id: str
    node_id: str
    gpu_index: int
    avg_iteration_time_ms: float
    slowdown_factor: float  # how much slower than the mean

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RayStragglerInfo:
        return cls(
            actor_id=str(data.get("actor_id", "")),
            node_id=str(data.get("node_id", "")),
            gpu_index=int(data.get("gpu_index", -1)),  # type: ignore[arg-type]
            avg_iteration_time_ms=float(data.get("avg_iteration_time_ms", 0.0)),  # type: ignore[arg-type]
            slowdown_factor=float(data.get("slowdown_factor", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class RayProfileResult:
    """Aggregated output of a Ray Train profiling session."""

    actors: List[RayActorMetrics]
    actor_gpu_map: List[RayActorGPUMapping]
    stragglers: List[RayStragglerInfo]
    num_workers: int
    num_nodes: int
    avg_iteration_time_ms: float
    iteration_time_std_ms: float
    total_training_time_s: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "actors": [a.to_dict() for a in self.actors],
            "actor_gpu_map": [m.to_dict() for m in self.actor_gpu_map],
            "stragglers": [s.to_dict() for s in self.stragglers],
            "num_workers": self.num_workers,
            "num_nodes": self.num_nodes,
            "avg_iteration_time_ms": self.avg_iteration_time_ms,
            "iteration_time_std_ms": self.iteration_time_std_ms,
            "total_training_time_s": self.total_training_time_s,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RayProfileResult:
        raw_actors = data.get("actors", [])
        actors = [
            RayActorMetrics.from_dict(a)  # type: ignore[arg-type]
            for a in raw_actors  # type: ignore[union-attr]
        ]
        raw_map = data.get("actor_gpu_map", [])
        actor_gpu_map = [
            RayActorGPUMapping.from_dict(m)  # type: ignore[arg-type]
            for m in raw_map  # type: ignore[union-attr]
        ]
        raw_stragglers = data.get("stragglers", [])
        stragglers = [
            RayStragglerInfo.from_dict(s)  # type: ignore[arg-type]
            for s in raw_stragglers  # type: ignore[union-attr]
        ]
        raw_recs = data.get("recommendations", [])
        recommendations: List[str] = (
            [str(r) for r in raw_recs]  # type: ignore[union-attr]
            if isinstance(raw_recs, list)
            else []
        )
        return cls(
            actors=actors,
            actor_gpu_map=actor_gpu_map,
            stragglers=stragglers,
            num_workers=int(data.get("num_workers", 0)),  # type: ignore[arg-type]
            num_nodes=int(data.get("num_nodes", 0)),  # type: ignore[arg-type]
            avg_iteration_time_ms=float(data.get("avg_iteration_time_ms", 0.0)),  # type: ignore[arg-type]
            iteration_time_std_ms=float(data.get("iteration_time_std_ms", 0.0)),  # type: ignore[arg-type]
            total_training_time_s=float(data.get("total_training_time_s", 0.0)),  # type: ignore[arg-type]
            recommendations=recommendations,
        )


# ============================================================================
# Ray Profiler
# ============================================================================


class RayProfiler:
    """Profile Ray Train distributed training jobs.

    Scrapes Ray runtime metrics, maps actors to GPUs, and detects
    stragglers in distributed Ray training.

    Usage::

        profiler = RayProfiler()
        profiler.start()
        # ... Ray Train job runs ...
        result = profiler.stop()
        for straggler in result.stragglers:
            print(straggler.actor_id, straggler.slowdown_factor)
    """

    def __init__(
        self,
        dashboard_url: Optional[str] = None,
    ) -> None:
        self._dashboard_url = dashboard_url
        self._active: bool = False
        self._start_time: float = 0.0
        self._actor_metrics: List[RayActorMetrics] = []
        self._actor_gpu_map: List[RayActorGPUMapping] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start profiling Ray Train actors."""
        if self._active:
            logger.warning("RayProfiler is already active; ignoring duplicate start().")
            return

        self._actor_metrics = []
        self._actor_gpu_map = []
        self._start_time = time.monotonic()

        if not _ray_available:
            logger.warning(
                "ray is not installed. Ray Train profiling is disabled; "
                "stop() will return empty results."
            )

        self._active = True

        if _ray_available and ray.is_initialized():
            self._scrape_initial_state()

        logger.debug("RayProfiler started.")

    def stop(self) -> RayProfileResult:
        """Stop profiling and return the analysed result."""
        if not self._active:
            logger.warning("RayProfiler.stop() called but profiler is not active.")
            return self._empty_result()

        self._active = False
        duration = time.monotonic() - self._start_time if self._start_time else 0.0

        if _ray_available and ray.is_initialized():
            self._scrape_final_state()

        actors = list(self._actor_metrics)
        gpu_map = list(self._actor_gpu_map)
        stragglers = self.detect_stragglers(actors)
        recommendations = self._generate_recommendations(actors, stragglers)

        avg_iter, std_iter = self._compute_iteration_stats(actors)
        num_nodes = len(set(a.node_id for a in actors)) if actors else 0

        return RayProfileResult(
            actors=actors,
            actor_gpu_map=gpu_map,
            stragglers=stragglers,
            num_workers=len(actors),
            num_nodes=num_nodes,
            avg_iteration_time_ms=round(avg_iter, 3),
            iteration_time_std_ms=round(std_iter, 3),
            total_training_time_s=round(duration, 3),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Actor-GPU mapping
    # ------------------------------------------------------------------

    def map_actors_to_gpus(self) -> List[RayActorGPUMapping]:
        """Query Ray to discover actor-to-GPU mappings.

        Returns the current mapping of actors to GPU devices. Each actor
        in a Ray Train job is typically assigned one GPU via resource
        scheduling.
        """
        if not _ray_available or not ray.is_initialized():
            return []

        mappings: List[RayActorGPUMapping] = []
        try:
            actors = ray.state.actors()  # type: ignore[attr-defined]
            nodes = ray.nodes()  # type: ignore[attr-defined]

            node_map: Dict[str, str] = {}
            for node in nodes:
                node_map[node.get("NodeID", "")] = node.get("NodeManagerAddress", "")

            for actor_id, info in actors.items():
                if info.get("State") != "ALIVE":
                    continue

                resources = info.get("Resources", {})
                gpu_count = resources.get("GPU", 0)
                if gpu_count <= 0:
                    continue

                node_id = info.get("NodeID", "")
                node_ip = node_map.get(node_id, "")

                # Ray assigns GPU indices via CUDA_VISIBLE_DEVICES
                gpu_index = int(resources.get("GPU_group_0_index", 0))

                mappings.append(RayActorGPUMapping(
                    actor_id=str(actor_id),
                    node_ip=node_ip,
                    node_id=node_id,
                    gpu_index=gpu_index,
                    gpu_name="",  # filled later if nvidia-smi is available
                ))
        except Exception:
            logger.debug("Failed to map Ray actors to GPUs.", exc_info=True)

        self._actor_gpu_map = mappings
        return mappings

    # ------------------------------------------------------------------
    # Metric scraping
    # ------------------------------------------------------------------

    def scrape_ray_metrics(self) -> List[RayActorMetrics]:
        """Scrape current metrics from Ray runtime for all Train actors.

        Uses ``ray.state.actors()`` to discover workers and collects
        available performance metrics.
        """
        if not _ray_available or not ray.is_initialized():
            return []

        metrics: List[RayActorMetrics] = []
        ts = time.time()

        try:
            actors = ray.state.actors()  # type: ignore[attr-defined]
            for actor_id, info in actors.items():
                if info.get("State") != "ALIVE":
                    continue

                resources = info.get("Resources", {})
                if resources.get("GPU", 0) <= 0:
                    continue

                metrics.append(RayActorMetrics(
                    actor_id=str(actor_id),
                    node_id=info.get("NodeID", ""),
                    gpu_index=int(resources.get("GPU_group_0_index", 0)),
                    pid=int(info.get("Pid", 0)),
                    avg_iteration_time_ms=0.0,
                    total_iterations=0,
                    gpu_utilization_pct=0.0,
                    gpu_memory_used_mb=0.0,
                    status="running" if info.get("State") == "ALIVE" else "unknown",
                    timestamp=ts,
                ))
        except Exception:
            logger.debug("Failed to scrape Ray metrics.", exc_info=True)

        return metrics

    # ------------------------------------------------------------------
    # Straggler detection
    # ------------------------------------------------------------------

    def detect_stragglers(
        self,
        actors: Optional[List[RayActorMetrics]] = None,
    ) -> List[RayStragglerInfo]:
        """Detect straggler actors in the Ray Train job.

        An actor is a straggler if its average iteration time exceeds
        ``mean + _STRAGGLER_SIGMA_THRESHOLD * std`` across all actors.

        Parameters
        ----------
        actors:
            Actor metrics to analyze. Defaults to internally collected.

        Returns
        -------
        list[RayStragglerInfo]
            Detected stragglers sorted by slowdown factor (worst first).
        """
        if actors is None:
            actors = list(self._actor_metrics)

        # Filter to actors with valid iteration times.
        active = [a for a in actors if a.avg_iteration_time_ms > 0]

        if len(active) < _MIN_ACTORS_FOR_STRAGGLER:
            return []

        times = [a.avg_iteration_time_ms for a in active]
        mean_time = sum(times) / len(times)

        if mean_time <= 0:
            return []

        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5

        if std_time <= 0:
            return []

        threshold = mean_time + _STRAGGLER_SIGMA_THRESHOLD * std_time

        stragglers: List[RayStragglerInfo] = []
        for actor in active:
            if actor.avg_iteration_time_ms > threshold:
                slowdown = actor.avg_iteration_time_ms / mean_time
                stragglers.append(RayStragglerInfo(
                    actor_id=actor.actor_id,
                    node_id=actor.node_id,
                    gpu_index=actor.gpu_index,
                    avg_iteration_time_ms=actor.avg_iteration_time_ms,
                    slowdown_factor=round(slowdown, 3),
                ))

        stragglers.sort(key=lambda s: s.slowdown_factor, reverse=True)
        return stragglers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrape_initial_state(self) -> None:
        """Capture actor state at profiling start."""
        self._actor_gpu_map = self.map_actors_to_gpus()

    def _scrape_final_state(self) -> None:
        """Capture actor state at profiling end."""
        self._actor_metrics = self.scrape_ray_metrics()

    @staticmethod
    def _compute_iteration_stats(
        actors: List[RayActorMetrics],
    ) -> Tuple[float, float]:
        """Compute mean and std of iteration times across actors."""
        times = [a.avg_iteration_time_ms for a in actors if a.avg_iteration_time_ms > 0]
        if not times:
            return 0.0, 0.0

        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        return mean_time, variance ** 0.5

    @staticmethod
    def _generate_recommendations(
        actors: List[RayActorMetrics],
        stragglers: List[RayStragglerInfo],
    ) -> List[str]:
        """Generate recommendations based on Ray profiling data."""
        recommendations: List[str] = []

        if stragglers:
            straggler_nodes = set(s.node_id for s in stragglers)
            recommendations.append(
                f"Detected {len(stragglers)} straggler worker(s) on "
                f"{len(straggler_nodes)} node(s). Check for thermal throttling, "
                f"slow interconnects, or uneven data distribution."
            )

        if actors:
            low_util = [a for a in actors if 0 < a.gpu_utilization_pct < 50]
            if low_util:
                recommendations.append(
                    f"{len(low_util)} worker(s) have low GPU utilization (<50%). "
                    f"Consider increasing per-worker batch size or enabling "
                    f"torch.compile() on the training function."
                )

        if not actors:
            recommendations.append(
                "No Ray Train actors detected. Ensure Ray is initialized "
                "and a Train job is running before profiling."
            )

        return recommendations

    @staticmethod
    def _empty_result() -> RayProfileResult:
        """Return an empty result."""
        return RayProfileResult(
            actors=[],
            actor_gpu_map=[],
            stragglers=[],
            num_workers=0,
            num_nodes=0,
            avg_iteration_time_ms=0.0,
            iteration_time_std_ms=0.0,
            total_training_time_s=0.0,
            recommendations=[],
        )
