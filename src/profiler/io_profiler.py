"""Storage I/O profiler for ML training workloads.

Profiles the three main I/O-bound operations in ML pipelines:
1. Dataset read speed — how fast data can be loaded from disk
2. Checkpoint write speed — how fast model checkpoints are saved
3. Model save/load time — full model serialization round-trip

Reports throughput in MB/s, identifies I/O bottlenecks, and generates
recommendations for faster storage backends.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import torch
# ---------------------------------------------------------------------------
_torch_available: bool = False
try:
    import torch  # type: ignore[import-untyped]
    _torch_available = True
except ImportError:
    pass


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class IOOperation:
    """Measurement of a single I/O operation."""

    operation: str  # "read", "write", "save", "load"
    path: str
    size_mb: float
    duration_sec: float
    throughput_mbps: float
    timestamp: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> IOOperation:
        return cls(
            operation=str(data.get("operation", "")),
            path=str(data.get("path", "")),
            size_mb=float(data.get("size_mb", 0.0)),  # type: ignore[arg-type]
            duration_sec=float(data.get("duration_sec", 0.0)),  # type: ignore[arg-type]
            throughput_mbps=float(data.get("throughput_mbps", 0.0)),  # type: ignore[arg-type]
            timestamp=float(data.get("timestamp", 0.0)),  # type: ignore[arg-type]
        )


@dataclass
class IOProfileResult:
    """Aggregated output of an I/O profiling session."""

    operations: List[IOOperation]
    avg_read_throughput_mbps: float
    avg_write_throughput_mbps: float
    avg_save_throughput_mbps: float
    avg_load_throughput_mbps: float
    total_io_time_sec: float
    is_io_bottleneck: bool
    recommendations: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "avg_read_throughput_mbps": self.avg_read_throughput_mbps,
            "avg_write_throughput_mbps": self.avg_write_throughput_mbps,
            "avg_save_throughput_mbps": self.avg_save_throughput_mbps,
            "avg_load_throughput_mbps": self.avg_load_throughput_mbps,
            "total_io_time_sec": self.total_io_time_sec,
            "is_io_bottleneck": self.is_io_bottleneck,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> IOProfileResult:
        raw_ops = data.get("operations", [])
        ops = [
            IOOperation.from_dict(o)  # type: ignore[arg-type]
            for o in raw_ops  # type: ignore[union-attr]
        ] if isinstance(raw_ops, list) else []

        raw_recs = data.get("recommendations", [])
        recs = [str(r) for r in raw_recs] if isinstance(raw_recs, list) else []  # type: ignore[union-attr]

        return cls(
            operations=ops,
            avg_read_throughput_mbps=float(data.get("avg_read_throughput_mbps", 0.0)),  # type: ignore[arg-type]
            avg_write_throughput_mbps=float(data.get("avg_write_throughput_mbps", 0.0)),  # type: ignore[arg-type]
            avg_save_throughput_mbps=float(data.get("avg_save_throughput_mbps", 0.0)),  # type: ignore[arg-type]
            avg_load_throughput_mbps=float(data.get("avg_load_throughput_mbps", 0.0)),  # type: ignore[arg-type]
            total_io_time_sec=float(data.get("total_io_time_sec", 0.0)),  # type: ignore[arg-type]
            is_io_bottleneck=bool(data.get("is_io_bottleneck", False)),
            recommendations=recs,
        )


# ============================================================================
# I/O Profiler
# ============================================================================

# Throughput thresholds (MB/s) — below these, I/O is considered slow.
_SLOW_READ_THRESHOLD_MBPS: float = 200.0
_SLOW_WRITE_THRESHOLD_MBPS: float = 100.0


class IOProfiler:
    """Profile storage I/O operations in ML training pipelines.

    Provides methods to benchmark:
    - Raw file read/write throughput
    - PyTorch model checkpoint save/load
    - Custom I/O operations via a callback

    Usage::

        profiler = IOProfiler()
        profiler.profile_file_read("/data/train.bin")
        profiler.profile_file_write("/checkpoints/model.pt", size_mb=500)
        result = profiler.get_results()
        print(result.avg_read_throughput_mbps)
    """

    def __init__(self) -> None:
        self._operations: List[IOOperation] = []

    # ------------------------------------------------------------------
    # Public API: file I/O benchmarks
    # ------------------------------------------------------------------

    def profile_file_read(
        self, path: str, block_size: int = 1024 * 1024,
    ) -> IOOperation:
        """Measure raw read throughput for a file.

        Parameters
        ----------
        path:
            Path to the file to read.
        block_size:
            Read block size in bytes.

        Returns
        -------
        IOOperation
        """
        if not os.path.isfile(path):
            logger.warning("File not found for read profiling: %s", path)
            op = IOOperation(
                operation="read", path=path, size_mb=0.0,
                duration_sec=0.0, throughput_mbps=0.0,
                timestamp=time.time(),
            )
            self._operations.append(op)
            return op

        file_size = os.path.getsize(path)
        size_mb = file_size / (1024 * 1024)

        # Drop OS page cache if possible (best effort).
        start = time.perf_counter()
        with open(path, "rb") as f:
            while f.read(block_size):
                pass
        duration = time.perf_counter() - start

        throughput = size_mb / duration if duration > 0 else 0.0

        op = IOOperation(
            operation="read",
            path=path,
            size_mb=round(size_mb, 3),
            duration_sec=round(duration, 6),
            throughput_mbps=round(throughput, 2),
            timestamp=time.time(),
        )
        self._operations.append(op)
        return op

    def profile_file_write(
        self,
        path: str,
        size_mb: float = 100.0,
        block_size: int = 1024 * 1024,
    ) -> IOOperation:
        """Measure raw write throughput.

        Writes ``size_mb`` of data to ``path`` and measures throughput.

        Parameters
        ----------
        path:
            Path to write to. Parent directory must exist.
        size_mb:
            Amount of data to write in MB.
        block_size:
            Write block size in bytes.
        """
        total_bytes = int(size_mb * 1024 * 1024)
        data_block = b"\x00" * block_size

        start = time.perf_counter()
        try:
            with open(path, "wb") as f:
                written = 0
                while written < total_bytes:
                    chunk = min(block_size, total_bytes - written)
                    f.write(data_block[:chunk])
                    written += chunk
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.warning("Write profiling failed: %s", e)
            op = IOOperation(
                operation="write", path=path, size_mb=size_mb,
                duration_sec=0.0, throughput_mbps=0.0,
                timestamp=time.time(),
            )
            self._operations.append(op)
            return op

        duration = time.perf_counter() - start
        throughput = size_mb / duration if duration > 0 else 0.0

        op = IOOperation(
            operation="write",
            path=path,
            size_mb=round(size_mb, 3),
            duration_sec=round(duration, 6),
            throughput_mbps=round(throughput, 2),
            timestamp=time.time(),
        )
        self._operations.append(op)
        return op

    # ------------------------------------------------------------------
    # Public API: model checkpoint profiling
    # ------------------------------------------------------------------

    def profile_checkpoint_save(
        self,
        state_dict: Any,
        path: str,
    ) -> IOOperation:
        """Profile saving a PyTorch state dict to disk.

        Parameters
        ----------
        state_dict:
            The state dict to save (e.g., ``model.state_dict()``).
        path:
            Destination file path.
        """
        if not _torch_available:
            logger.warning("torch not available; skipping checkpoint save profiling.")
            op = IOOperation(
                operation="save", path=path, size_mb=0.0,
                duration_sec=0.0, throughput_mbps=0.0,
                timestamp=time.time(),
            )
            self._operations.append(op)
            return op

        start = time.perf_counter()
        torch.save(state_dict, path)
        duration = time.perf_counter() - start

        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.isfile(path) else 0.0
        throughput = size_mb / duration if duration > 0 else 0.0

        op = IOOperation(
            operation="save",
            path=path,
            size_mb=round(size_mb, 3),
            duration_sec=round(duration, 6),
            throughput_mbps=round(throughput, 2),
            timestamp=time.time(),
        )
        self._operations.append(op)
        return op

    def profile_checkpoint_load(self, path: str) -> IOOperation:
        """Profile loading a PyTorch checkpoint from disk.

        Parameters
        ----------
        path:
            Path to the checkpoint file.
        """
        if not _torch_available:
            logger.warning("torch not available; skipping checkpoint load profiling.")
            op = IOOperation(
                operation="load", path=path, size_mb=0.0,
                duration_sec=0.0, throughput_mbps=0.0,
                timestamp=time.time(),
            )
            self._operations.append(op)
            return op

        if not os.path.isfile(path):
            logger.warning("Checkpoint file not found: %s", path)
            op = IOOperation(
                operation="load", path=path, size_mb=0.0,
                duration_sec=0.0, throughput_mbps=0.0,
                timestamp=time.time(),
            )
            self._operations.append(op)
            return op

        size_mb = os.path.getsize(path) / (1024 * 1024)

        start = time.perf_counter()
        torch.load(path, map_location="cpu", weights_only=True)
        duration = time.perf_counter() - start

        throughput = size_mb / duration if duration > 0 else 0.0

        op = IOOperation(
            operation="load",
            path=path,
            size_mb=round(size_mb, 3),
            duration_sec=round(duration, 6),
            throughput_mbps=round(throughput, 2),
            timestamp=time.time(),
        )
        self._operations.append(op)
        return op

    # ------------------------------------------------------------------
    # Public API: custom I/O profiling
    # ------------------------------------------------------------------

    def profile_operation(
        self,
        name: str,
        fn: Callable[[], None],
        size_mb: float,
        path: str = "",
    ) -> IOOperation:
        """Profile an arbitrary I/O operation.

        Parameters
        ----------
        name:
            Operation name (e.g. "dataset_read", "export").
        fn:
            Callable that performs the I/O operation.
        size_mb:
            Known size of data involved (MB).
        path:
            Associated file path (for reporting).
        """
        start = time.perf_counter()
        fn()
        duration = time.perf_counter() - start

        throughput = size_mb / duration if duration > 0 else 0.0

        op = IOOperation(
            operation=name,
            path=path,
            size_mb=round(size_mb, 3),
            duration_sec=round(duration, 6),
            throughput_mbps=round(throughput, 2),
            timestamp=time.time(),
        )
        self._operations.append(op)
        return op

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> IOProfileResult:
        """Aggregate all recorded operations into a result."""
        ops = list(self._operations)
        if not ops:
            return self._empty_result()

        def _avg_throughput(op_type: str) -> float:
            matching = [o for o in ops if o.operation == op_type and o.throughput_mbps > 0]
            if not matching:
                return 0.0
            return sum(o.throughput_mbps for o in matching) / len(matching)

        avg_read = _avg_throughput("read")
        avg_write = _avg_throughput("write")
        avg_save = _avg_throughput("save")
        avg_load = _avg_throughput("load")

        total_time = sum(o.duration_sec for o in ops)

        is_bottleneck = self._detect_bottleneck(ops)
        recs = self._generate_recommendations(ops, avg_read, avg_write, avg_save, avg_load)

        return IOProfileResult(
            operations=ops,
            avg_read_throughput_mbps=round(avg_read, 2),
            avg_write_throughput_mbps=round(avg_write, 2),
            avg_save_throughput_mbps=round(avg_save, 2),
            avg_load_throughput_mbps=round(avg_load, 2),
            total_io_time_sec=round(total_time, 3),
            is_io_bottleneck=is_bottleneck,
            recommendations=recs,
        )

    def reset(self) -> None:
        """Clear all recorded operations."""
        self._operations = []

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_bottleneck(ops: List[IOOperation]) -> bool:
        """Determine if I/O is a bottleneck based on throughput thresholds."""
        read_ops = [o for o in ops if o.operation == "read" and o.throughput_mbps > 0]
        write_ops = [o for o in ops if o.operation in ("write", "save") and o.throughput_mbps > 0]

        if read_ops:
            avg_read = sum(o.throughput_mbps for o in read_ops) / len(read_ops)
            if avg_read < _SLOW_READ_THRESHOLD_MBPS:
                return True

        if write_ops:
            avg_write = sum(o.throughput_mbps for o in write_ops) / len(write_ops)
            if avg_write < _SLOW_WRITE_THRESHOLD_MBPS:
                return True

        return False

    @staticmethod
    def _generate_recommendations(
        ops: List[IOOperation],
        avg_read: float,
        avg_write: float,
        avg_save: float,
        avg_load: float,
    ) -> List[str]:
        """Generate I/O optimization recommendations."""
        recs: List[str] = []

        if avg_read > 0 and avg_read < _SLOW_READ_THRESHOLD_MBPS:
            recs.append(
                f"Read throughput is {avg_read:.0f} MB/s (below {_SLOW_READ_THRESHOLD_MBPS:.0f} MB/s threshold). "
                f"Consider using NVMe SSD storage, memory-mapped files, or a faster "
                f"data format (WebDataset, FFCV)."
            )

        if avg_write > 0 and avg_write < _SLOW_WRITE_THRESHOLD_MBPS:
            recs.append(
                f"Write throughput is {avg_write:.0f} MB/s (below {_SLOW_WRITE_THRESHOLD_MBPS:.0f} MB/s threshold). "
                f"Consider using local NVMe storage for checkpoints, or async "
                f"checkpoint saving to avoid blocking training."
            )

        if avg_save > 0 and avg_save < _SLOW_WRITE_THRESHOLD_MBPS:
            recs.append(
                f"Checkpoint save throughput is {avg_save:.0f} MB/s. "
                f"Consider using torch.save with async I/O, or saving to a "
                f"RAM disk and copying to persistent storage in the background."
            )

        if avg_load > 0 and avg_load < _SLOW_READ_THRESHOLD_MBPS:
            recs.append(
                f"Checkpoint load throughput is {avg_load:.0f} MB/s. "
                f"Consider using safetensors format for faster model loading, "
                f"or memory-mapping the checkpoint file."
            )

        # Check for large total I/O time.
        total_time = sum(o.duration_sec for o in ops)
        if total_time > 60:
            recs.append(
                f"Total I/O time is {total_time:.1f}s. I/O operations are consuming "
                f"significant wall-clock time. Consider overlapping I/O with "
                f"computation using background threads or async I/O."
            )

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> IOProfileResult:
        return IOProfileResult(
            operations=[],
            avg_read_throughput_mbps=0.0,
            avg_write_throughput_mbps=0.0,
            avg_save_throughput_mbps=0.0,
            avg_load_throughput_mbps=0.0,
            total_io_time_sec=0.0,
            is_io_bottleneck=False,
            recommendations=[],
        )
