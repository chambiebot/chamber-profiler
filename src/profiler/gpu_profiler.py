"""Core GPU profiling engine.

Collects GPU utilization, memory, power, temperature, SM activity, memory bandwidth,
and clock speed metrics at configurable intervals using a background daemon thread.

Supports two backends:
  1. pynvml (preferred) - direct NVML bindings for low-overhead collection
  2. nvidia-smi (fallback) - subprocess-based collection via the CLI tool

If neither backend is available, the profiler logs a warning and produces empty results.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import ClassVar, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import pynvml
# ---------------------------------------------------------------------------
_pynvml_available: bool = False
try:
    import pynvml  # type: ignore[import-untyped]

    _pynvml_available = True
    logger.debug("pynvml is available; will use NVML backend for GPU metrics.")
except ImportError:
    logger.debug("pynvml is not installed; will attempt nvidia-smi fallback.")

# ---------------------------------------------------------------------------
# Probe for nvidia-smi on the PATH
# ---------------------------------------------------------------------------
_nvidia_smi_available: bool = False
if not _pynvml_available:
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        _nvidia_smi_available = True
        logger.debug("nvidia-smi is available; will use CLI fallback for GPU metrics.")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.debug("nvidia-smi is not available on this system.")


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class GPUMetricSnapshot:
    """A single point-in-time measurement of one GPU's metrics."""

    timestamp: float
    gpu_index: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    power_watts: float
    temperature_c: float
    sm_activity_pct: float
    memory_bandwidth_pct: float
    clock_speed_mhz: float

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> GPUMetricSnapshot:
        return cls(**data)  # type: ignore[arg-type]


@dataclass
class ProfileResult:
    """Aggregated result of a profiling session."""

    duration_s: float
    snapshots: List[GPUMetricSnapshot]
    gpu_count: int

    # Computed summary statistics
    avg_utilization: float = 0.0
    peak_memory_mb: float = 0.0
    avg_power_watts: float = 0.0

    # Class-level sentinel used by from_dict when no snapshots are provided.
    _EMPTY: ClassVar[List[GPUMetricSnapshot]] = []

    def __post_init__(self) -> None:
        """Recompute derived statistics from the raw snapshots."""
        if self.snapshots:
            self.avg_utilization = sum(s.utilization_pct for s in self.snapshots) / len(
                self.snapshots
            )
            self.peak_memory_mb = max(s.memory_used_mb for s in self.snapshots)
            self.avg_power_watts = sum(s.power_watts for s in self.snapshots) / len(
                self.snapshots
            )

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "duration_s": self.duration_s,
            "gpu_count": self.gpu_count,
            "avg_utilization": self.avg_utilization,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_power_watts": self.avg_power_watts,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ProfileResult:
        raw_snapshots = data.get("snapshots", [])
        snapshots = [
            GPUMetricSnapshot.from_dict(s)
            for s in raw_snapshots  # type: ignore[union-attr]
        ]
        return cls(
            duration_s=float(data.get("duration_s", 0.0)),  # type: ignore[arg-type]
            snapshots=snapshots,
            gpu_count=int(data.get("gpu_count", 0)),  # type: ignore[arg-type]
        )


# ============================================================================
# GPU Profiler
# ============================================================================


class GPUProfiler:
    """Background-thread GPU metric collector.

    Usage::

        profiler = GPUProfiler(interval_ms=100, gpu_indices=[0, 1])
        profiler.start()
        # ... run workload ...
        result = profiler.stop()
        print(result.avg_utilization, result.peak_memory_mb)
    """

    def __init__(
        self,
        interval_ms: int = 100,
        gpu_indices: Optional[List[int]] = None,
    ) -> None:
        self._interval_s: float = interval_ms / 1000.0
        self._gpu_indices: Optional[List[int]] = gpu_indices

        self._snapshots: List[GPUMetricSnapshot] = []
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

        # Resolved GPU list (populated in start())
        self._resolved_indices: List[int] = []

        # Backend state
        self._backend: Optional[str] = None  # "pynvml" | "nvidia-smi" | None
        self._nvml_handles: Dict[int, object] = {}  # gpu_index -> nvmlDeviceHandle
        self._nvml_initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background metric collection."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("GPUProfiler is already running; ignoring duplicate start().")
            return

        self._snapshots = []
        self._stop_event.clear()

        # Initialise backend & resolve GPU indices
        self._init_backend()

        if self._backend is None:
            logger.warning(
                "No GPU monitoring backend available (pynvml not installed, "
                "nvidia-smi not found). Profiling will produce empty results."
            )

        self._start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._collect_metrics,
            name="gpu-profiler",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "GPU profiler started (interval=%.1f ms, gpus=%s, backend=%s).",
            self._interval_s * 1000,
            self._resolved_indices,
            self._backend,
        )

    def stop(self) -> ProfileResult:
        """Stop collection and return the aggregated :class:`ProfileResult`."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 5, 2.0))
            self._thread = None

        duration = time.monotonic() - self._start_time if self._start_time else 0.0

        self._shutdown_backend()

        with self._lock:
            snapshots = list(self._snapshots)

        return ProfileResult(
            duration_s=duration,
            snapshots=snapshots,
            gpu_count=len(self._resolved_indices),
        )

    def get_live_metrics(self) -> Dict[int, GPUMetricSnapshot]:
        """Return the most recent snapshot for each tracked GPU.

        Returns a dict mapping ``gpu_index`` to its latest
        :class:`GPUMetricSnapshot`, or an empty dict if no data has been
        collected yet.
        """
        with self._lock:
            latest: Dict[int, GPUMetricSnapshot] = {}
            # Walk backwards so we find the most recent for each GPU first.
            for snap in reversed(self._snapshots):
                if snap.gpu_index not in latest:
                    latest[snap.gpu_index] = snap
                if len(latest) == len(self._resolved_indices):
                    break
            return latest

    # ------------------------------------------------------------------
    # Background collection loop
    # ------------------------------------------------------------------

    def _collect_metrics(self) -> None:
        """Run in the background thread, collecting snapshots at the configured interval."""
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            for idx in self._resolved_indices:
                snapshot = self._read_gpu_metrics(idx)
                if snapshot is not None:
                    with self._lock:
                        self._snapshots.append(snapshot)
            # Sleep for the remainder of the interval (account for collection time).
            elapsed = time.monotonic() - loop_start
            sleep_time = self._interval_s - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    # ------------------------------------------------------------------
    # Metric reading (dispatch)
    # ------------------------------------------------------------------

    def _read_gpu_metrics(self, gpu_index: int) -> Optional[GPUMetricSnapshot]:
        """Read metrics for *gpu_index*, trying pynvml first then nvidia-smi."""
        try:
            if self._backend == "pynvml":
                return self._read_metrics_pynvml(gpu_index)
            elif self._backend == "nvidia-smi":
                snapshots = self._read_metrics_nvidia_smi()
                for snap in snapshots:
                    if snap.gpu_index == gpu_index:
                        return snap
                return None
            else:
                return None
        except Exception:
            logger.debug(
                "Failed to read GPU metrics for index %d.", gpu_index, exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # pynvml backend
    # ------------------------------------------------------------------

    def _read_metrics_pynvml(self, gpu_index: int) -> GPUMetricSnapshot:
        """Read metrics for one GPU via the pynvml library."""
        handle = self._nvml_handles[gpu_index]
        ts = time.time()

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            power_w = power_mw / 1000.0
        except pynvml.NVMLError:
            power_w = 0.0

        try:
            temperature = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except pynvml.NVMLError:
            temperature = 0.0

        try:
            clock_mhz = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
        except pynvml.NVMLError:
            clock_mhz = 0.0

        return GPUMetricSnapshot(
            timestamp=ts,
            gpu_index=gpu_index,
            utilization_pct=float(utilization.gpu),
            memory_used_mb=mem_info.used / (1024 * 1024),
            memory_total_mb=mem_info.total / (1024 * 1024),
            power_watts=power_w,
            temperature_c=temperature,
            sm_activity_pct=float(utilization.gpu),  # NVML reports SM-level as "gpu"
            memory_bandwidth_pct=float(utilization.memory),
            clock_speed_mhz=clock_mhz,
        )

    # ------------------------------------------------------------------
    # nvidia-smi fallback backend
    # ------------------------------------------------------------------

    def _read_metrics_nvidia_smi(self) -> List[GPUMetricSnapshot]:
        """Read metrics for *all* GPUs via the nvidia-smi CLI.

        Returns a list of :class:`GPUMetricSnapshot`, one per GPU reported
        by ``nvidia-smi``.  On failure returns an empty list.
        """
        query_fields = (
            "index,"
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "power.draw,"
            "temperature.gpu,"
            "utilization.gpu,"    # SM activity (best approximation via CLI)
            "utilization.memory,"
            "clocks.current.sm"
        )
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query_fields}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("nvidia-smi invocation failed: %s", exc)
            return []

        if result.returncode != 0:
            logger.debug("nvidia-smi returned non-zero exit code %d.", result.returncode)
            return []

        ts = time.time()
        snapshots: List[GPUMetricSnapshot] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                logger.debug("Skipping malformed nvidia-smi line: %r", line)
                continue
            try:
                snapshots.append(
                    GPUMetricSnapshot(
                        timestamp=ts,
                        gpu_index=int(parts[0]),
                        utilization_pct=_safe_float(parts[1]),
                        memory_used_mb=_safe_float(parts[2]),
                        memory_total_mb=_safe_float(parts[3]),
                        power_watts=_safe_float(parts[4]),
                        temperature_c=_safe_float(parts[5]),
                        sm_activity_pct=_safe_float(parts[6]),
                        memory_bandwidth_pct=_safe_float(parts[7]),
                        clock_speed_mhz=_safe_float(parts[8]),
                    )
                )
            except (ValueError, IndexError) as exc:
                logger.debug("Failed to parse nvidia-smi row: %s (%s)", line, exc)

        return snapshots

    # ------------------------------------------------------------------
    # Backend lifecycle helpers
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        """Initialise the best available monitoring backend and resolve GPU indices."""
        # --- try pynvml first ---
        if _pynvml_available:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count == 0:
                    logger.warning("pynvml reports 0 GPUs.")
                    pynvml.nvmlShutdown()
                    self._nvml_initialized = False
                else:
                    self._resolved_indices = self._resolve_indices(device_count)
                    self._nvml_handles = {}
                    for idx in self._resolved_indices:
                        self._nvml_handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self._backend = "pynvml"
                    logger.debug(
                        "pynvml backend initialised with %d GPU(s).", len(self._resolved_indices)
                    )
                    return
            except Exception:
                logger.debug("pynvml initialisation failed.", exc_info=True)
                if self._nvml_initialized:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass
                    self._nvml_initialized = False

        # --- try nvidia-smi fallback ---
        if _nvidia_smi_available or not _pynvml_available:
            gpu_count = self._probe_gpu_count_nvidia_smi()
            if gpu_count > 0:
                self._resolved_indices = self._resolve_indices(gpu_count)
                self._backend = "nvidia-smi"
                logger.debug(
                    "nvidia-smi backend initialised with %d GPU(s).",
                    len(self._resolved_indices),
                )
                return

        # --- no backend ---
        self._backend = None
        self._resolved_indices = []

    def _shutdown_backend(self) -> None:
        """Release backend resources."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                logger.debug("pynvml shutdown failed.", exc_info=True)
            self._nvml_initialized = False
        self._nvml_handles = {}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _resolve_indices(self, device_count: int) -> List[int]:
        """Return the list of GPU indices to monitor.

        If the caller specified ``gpu_indices`` at construction time, validate
        them against *device_count*; otherwise return ``[0 .. device_count-1]``.
        """
        if self._gpu_indices is not None:
            valid = [i for i in self._gpu_indices if 0 <= i < device_count]
            if len(valid) < len(self._gpu_indices):
                skipped = set(self._gpu_indices) - set(valid)
                logger.warning(
                    "Requested GPU indices %s are out of range (device_count=%d); skipping.",
                    skipped,
                    device_count,
                )
            return valid
        return list(range(device_count))

    @staticmethod
    def _probe_gpu_count_nvidia_smi() -> int:
        """Use nvidia-smi to determine how many GPUs are present."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                return len(lines)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_float(value: str) -> float:
    """Parse a float from an nvidia-smi CSV field, returning 0.0 on failure.

    nvidia-smi sometimes emits ``[N/A]`` or empty strings for unsupported
    queries; this helper handles those gracefully.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
