"""CUDA kernel analysis module.

Traces GPU kernel execution via PyTorch's built-in profiler, classifies each
kernel into a semantic category (gemm, conv, attention, ...), estimates
occupancy, and surfaces the slowest and most inefficient kernels so users can
prioritize optimization work.

If PyTorch is not installed the tracer logs a warning and produces empty
results -- the rest of the profiling suite keeps working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch
# ---------------------------------------------------------------------------
_torch_available: bool = False
try:
    import torch  # type: ignore[import-untyped]
    from torch.profiler import profile as _torch_profile  # type: ignore[import-untyped]
    from torch.profiler import ProfilerActivity  # type: ignore[import-untyped]

    _torch_available = True
    logger.debug("torch is available; CUDA kernel tracing is supported.")
except ImportError:
    logger.debug("torch is not installed; kernel tracing will produce empty results.")

# ---------------------------------------------------------------------------
# Valid kernel categories
# ---------------------------------------------------------------------------
VALID_CATEGORIES: Tuple[str, ...] = (
    "gemm",
    "conv",
    "attention",
    "elementwise",
    "reduction",
    "memory",
    "communication",
    "other",
)

# ---------------------------------------------------------------------------
# Category classification patterns
# ---------------------------------------------------------------------------
# Each entry is (compiled_regex, category).  Order matters -- first match wins.
_CATEGORY_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # GEMM / matrix-multiply
    (re.compile(r"gemm|cutlass|cublas|cublasLt|wmma|mma_|sgemm|dgemm|hgemm", re.IGNORECASE), "gemm"),
    # Convolution
    (re.compile(r"conv|cudnn|winograd|im2col|col2im", re.IGNORECASE), "conv"),
    # Attention / flash-attention
    (re.compile(r"attention|flash|fmha|softmax_warp|sdpa", re.IGNORECASE), "attention"),
    # Collective communication
    (re.compile(r"nccl|allreduce|allgather|reducescatter|broadcast|all_to_all|ncclKernel", re.IGNORECASE), "communication"),
    # Memory operations
    (re.compile(r"memcpy|memset|copy_kernel|fill_kernel|Memcpy|Memset", re.IGNORECASE), "memory"),
    # Reduction
    (re.compile(r"reduce|norm_kernel|layernorm|batchnorm|group_norm|softmax|log_softmax", re.IGNORECASE), "reduction"),
    # Element-wise
    (re.compile(r"elementwise|pointwise|vectorized|unrolled|activation|relu|gelu|silu|sigmoid|tanh|add_kernel|mul_kernel|fused_dropout", re.IGNORECASE), "elementwise"),
]

# Patterns used to extract a human-readable layer name from kernel annotations.
_LAYER_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"(?:nn\.Module:\s*|autograd::)(\S+)", re.IGNORECASE
)

# Maximum number of SMs on current-generation GPUs.  Used as a rough upper
# bound when estimating occupancy.  A100 = 108, H100 = 132.
_MAX_SM_COUNT: int = 132


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class KernelRecord:
    """A single recorded CUDA kernel invocation."""

    name: str
    duration_us: float
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    shared_memory_bytes: int
    device_index: int
    category: str
    layer_name: Optional[str] = None
    occupancy: Optional[float] = None

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            logger.warning(
                "KernelRecord created with unknown category %r; defaulting to 'other'.",
                self.category,
            )
            self.category = "other"

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Tuples become lists via asdict; convert back to list for JSON compat.
        data["grid_size"] = list(self.grid_size)
        data["block_size"] = list(self.block_size)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KernelRecord:
        data = dict(data)  # shallow copy to avoid mutating caller's dict
        data["grid_size"] = tuple(data.get("grid_size", (0, 0, 0)))
        data["block_size"] = tuple(data.get("block_size", (0, 0, 0)))
        return cls(
            name=str(data.get("name", "")),
            duration_us=float(data.get("duration_us", 0.0)),
            grid_size=data["grid_size"],
            block_size=data["block_size"],
            shared_memory_bytes=int(data.get("shared_memory_bytes", 0)),
            device_index=int(data.get("device_index", 0)),
            category=str(data.get("category", "other")),
            layer_name=data.get("layer_name"),
            occupancy=data.get("occupancy"),
        )


@dataclass
class KernelTraceResult:
    """Aggregated output of a kernel-tracing session."""

    kernels: List[KernelRecord]
    total_kernel_time_us: float
    top_kernels: List[KernelRecord]
    inefficient_kernels: List[KernelRecord]
    category_breakdown: Dict[str, float] = field(default_factory=dict)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernels": [k.to_dict() for k in self.kernels],
            "total_kernel_time_us": self.total_kernel_time_us,
            "top_kernels": [k.to_dict() for k in self.top_kernels],
            "inefficient_kernels": [k.to_dict() for k in self.inefficient_kernels],
            "category_breakdown": dict(self.category_breakdown),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KernelTraceResult:
        kernels = [KernelRecord.from_dict(k) for k in data.get("kernels", [])]
        top_kernels = [KernelRecord.from_dict(k) for k in data.get("top_kernels", [])]
        inefficient_kernels = [
            KernelRecord.from_dict(k) for k in data.get("inefficient_kernels", [])
        ]
        return cls(
            kernels=kernels,
            total_kernel_time_us=float(data.get("total_kernel_time_us", 0.0)),
            top_kernels=top_kernels,
            inefficient_kernels=inefficient_kernels,
            category_breakdown=dict(data.get("category_breakdown", {})),
        )


# ============================================================================
# Kernel Tracer
# ============================================================================


class KernelTracer:
    """Trace CUDA kernel execution via PyTorch's profiler.

    Usage::

        tracer = KernelTracer(top_k=10)
        tracer.start()
        # ... run workload ...
        result = tracer.stop()
        for k in result.top_kernels:
            print(k.name, k.duration_us, k.category)
    """

    # Occupancy thresholds used by get_inefficient_kernels.
    LOW_OCCUPANCY_THRESHOLD: ClassVar[float] = 0.25

    def __init__(self, top_k: int = 10) -> None:
        self._top_k: int = top_k
        self._profiler: Any = None  # torch.profiler.profile instance
        self._kernels: List[KernelRecord] = []
        self._active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin tracing CUDA kernels.

        Hooks into ``torch.profiler.profile``.  If torch is not available the
        call is a no-op and :meth:`stop` will return an empty
        :class:`KernelTraceResult`.
        """
        if self._active:
            logger.warning("KernelTracer is already active; ignoring duplicate start().")
            return

        self._kernels = []

        if not _torch_available:
            logger.warning(
                "torch is not installed. Kernel tracing is disabled; "
                "stop() will return empty results."
            )
            self._active = True
            return

        try:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            self._profiler = _torch_profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
            )
            self._profiler.__enter__()
            self._active = True
            logger.debug(
                "Kernel tracing started (top_k=%d, activities=%s).",
                self._top_k,
                [a.name for a in activities],
            )
        except Exception:
            logger.error("Failed to start PyTorch profiler.", exc_info=True)
            self._profiler = None
            self._active = True  # allow stop() to return empty result

    def stop(self) -> KernelTraceResult:
        """Stop tracing and return the analysed :class:`KernelTraceResult`."""
        if not self._active:
            logger.warning("KernelTracer.stop() called but tracer is not active.")
            return self._empty_result()

        self._active = False

        if self._profiler is not None:
            try:
                self._profiler.__exit__(None, None, None)
                events = self._profiler.key_averages()
                self._kernels = self._process_events(events)
            except Exception:
                logger.error(
                    "Failed to stop PyTorch profiler or process events.",
                    exc_info=True,
                )
                self._kernels = []
            finally:
                self._profiler = None

        # Build the result.
        total_time = sum(k.duration_us for k in self._kernels)
        top = self.get_top_kernels()
        inefficient = self.get_inefficient_kernels()
        breakdown = self._compute_category_breakdown(self._kernels)

        return KernelTraceResult(
            kernels=list(self._kernels),
            total_kernel_time_us=total_time,
            top_kernels=top,
            inefficient_kernels=inefficient,
            category_breakdown=breakdown,
        )

    def get_top_kernels(self, n: Optional[int] = None) -> List[KernelRecord]:
        """Return the *n* slowest kernels.  Defaults to ``self._top_k``."""
        count = n if n is not None else self._top_k
        sorted_kernels = sorted(self._kernels, key=lambda k: k.duration_us, reverse=True)
        return sorted_kernels[:count]

    def get_inefficient_kernels(self) -> List[KernelRecord]:
        """Return kernels that are likely inefficient.

        A kernel is considered inefficient if:
        - Its estimated occupancy is below :attr:`LOW_OCCUPANCY_THRESHOLD`, or
        - It is a ``memory`` category kernel (excessive memcpy/memset traffic
          often signals an optimisation opportunity).
        """
        inefficient: List[KernelRecord] = []
        for kernel in self._kernels:
            if kernel.category == "memory":
                inefficient.append(kernel)
            elif kernel.occupancy is not None and kernel.occupancy < self.LOW_OCCUPANCY_THRESHOLD:
                inefficient.append(kernel)
        return inefficient

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_events(self, events: Any) -> List[KernelRecord]:
        """Parse profiler ``key_averages`` into a list of :class:`KernelRecord`.

        Parameters
        ----------
        events:
            The iterable returned by ``torch.profiler.profile.key_averages()``.
        """
        records: List[KernelRecord] = []
        for event in events:
            try:
                # We only care about CUDA kernels.
                if not self._is_cuda_kernel(event):
                    continue

                name: str = getattr(event, "key", "") or getattr(event, "name", "")
                if not name:
                    continue

                # Duration in microseconds.  key_averages exposes several
                # timing attributes; prefer cuda_time_total (GPU wall time).
                duration_us: float = float(
                    getattr(event, "cuda_time_total", 0.0)
                    or getattr(event, "self_cuda_time_total", 0.0)
                )
                if duration_us <= 0:
                    continue

                grid_size = self._extract_tuple(event, "grid_size")
                block_size = self._extract_tuple(event, "block_size")
                shared_mem = int(getattr(event, "shared_memory", 0) or 0)
                device_index = int(getattr(event, "device_index", 0) or 0)

                category = self._categorize_kernel(name)
                layer_name = self._map_to_layer(event)

                record = KernelRecord(
                    name=name,
                    duration_us=duration_us,
                    grid_size=grid_size,
                    block_size=block_size,
                    shared_memory_bytes=shared_mem,
                    device_index=device_index,
                    category=category,
                    layer_name=layer_name,
                )
                record.occupancy = self._estimate_occupancy(record)
                records.append(record)
            except Exception:
                logger.debug(
                    "Skipping unparseable profiler event.", exc_info=True
                )

        return records

    # ------------------------------------------------------------------
    # Categorisation
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_kernel(name: str) -> str:
        """Classify a CUDA kernel name into a semantic category using regex
        pattern matching.

        Parameters
        ----------
        name:
            The raw kernel name as reported by the profiler (e.g.
            ``"volta_sgemm_128x64_nn"``).

        Returns
        -------
        str
            One of :data:`VALID_CATEGORIES`.
        """
        for pattern, category in _CATEGORY_PATTERNS:
            if pattern.search(name):
                return category
        return "other"

    # ------------------------------------------------------------------
    # Layer mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_layer(event: Any) -> Optional[str]:
        """Attempt to map a profiler event back to a model layer name.

        PyTorch's profiler annotates events with module hierarchy information
        when ``record_shapes=True`` and ``with_stack=True``.  This method
        inspects the available metadata to extract a human-readable layer
        name such as ``"transformer.encoder.layers.0.self_attn"``.
        """
        # Try the 'scope' or 'module_hierarchy' attributes first (available in
        # recent PyTorch versions).
        for attr in ("module_hierarchy", "scope", "stack"):
            value = getattr(event, attr, None)
            if value:
                # module_hierarchy / scope can be a string; stack is sometimes
                # a list.
                text = value if isinstance(value, str) else "\n".join(value)
                match = _LAYER_NAME_PATTERN.search(text)
                if match:
                    return match.group(1)

        # Fall back to the event name itself if it looks like a module path.
        name: str = getattr(event, "key", "") or getattr(event, "name", "")
        if "." in name and "::" not in name and not name.startswith("void "):
            return name

        return None

    # ------------------------------------------------------------------
    # Occupancy estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_occupancy(record: KernelRecord) -> Optional[float]:
        """Provide a rough occupancy estimate from grid / block dimensions.

        True occupancy depends on register usage, shared memory, and the
        target GPU architecture -- information we do not have at this point.
        Instead we compute a *launch-geometry heuristic*:

            occupancy ~ (total_threads / max_resident_threads)

        clamped to [0, 1].  This is coarse but useful for flagging very small
        launches that clearly under-utilise the GPU.
        """
        gx, gy, gz = record.grid_size
        bx, by, bz = record.block_size

        total_blocks = gx * gy * gz
        threads_per_block = bx * by * bz

        if total_blocks <= 0 or threads_per_block <= 0:
            return None

        total_threads = total_blocks * threads_per_block

        # Assume up to 2048 resident threads per SM (common since Volta) and
        # _MAX_SM_COUNT SMs.
        max_resident_threads = _MAX_SM_COUNT * 2048
        occupancy = min(total_threads / max_resident_threads, 1.0)
        return round(occupancy, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cuda_kernel(event: Any) -> bool:
        """Return ``True`` if the profiler event represents a CUDA kernel."""
        # torch.profiler events have a ``device_type`` attribute (int enum) or
        # a string-based ``device_type`` in older versions.
        device_type = getattr(event, "device_type", None)
        if device_type is not None:
            # DeviceType.CUDA == 1 in the C++ enum exposed by PyTorch.
            if isinstance(device_type, int) and device_type == 1:
                return True
            if hasattr(device_type, "value") and device_type.value == 1:
                return True

        # Fallback: check whether the event has non-zero CUDA time.
        cuda_time = getattr(event, "cuda_time_total", 0.0) or 0.0
        self_cuda_time = getattr(event, "self_cuda_time_total", 0.0) or 0.0
        if cuda_time > 0 or self_cuda_time > 0:
            return True

        return False

    @staticmethod
    def _extract_tuple(event: Any, attr: str) -> Tuple[int, int, int]:
        """Extract a 3-tuple of ints from an event attribute.

        The profiler may store grid/block sizes as tuples, lists, or nested
        objects.  This helper normalises to ``(int, int, int)``, returning
        ``(0, 0, 0)`` on failure.
        """
        raw = getattr(event, attr, None)
        if raw is None:
            return (0, 0, 0)

        try:
            if isinstance(raw, (tuple, list)) and len(raw) >= 3:
                return (int(raw[0]), int(raw[1]), int(raw[2]))

            # Some profiler versions expose .x / .y / .z attributes.
            if hasattr(raw, "x") and hasattr(raw, "y") and hasattr(raw, "z"):
                return (int(raw.x), int(raw.y), int(raw.z))
        except (TypeError, ValueError, AttributeError):
            pass

        return (0, 0, 0)

    @staticmethod
    def _compute_category_breakdown(
        kernels: Sequence[KernelRecord],
    ) -> Dict[str, float]:
        """Sum kernel durations per category."""
        breakdown: Dict[str, float] = {}
        for kernel in kernels:
            breakdown[kernel.category] = (
                breakdown.get(kernel.category, 0.0) + kernel.duration_us
            )
        return breakdown

    @staticmethod
    def _empty_result() -> KernelTraceResult:
        """Return an empty :class:`KernelTraceResult`."""
        return KernelTraceResult(
            kernels=[],
            total_kernel_time_us=0.0,
            top_kernels=[],
            inefficient_kernels=[],
            category_breakdown={},
        )
