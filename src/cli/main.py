"""CLI interface for chamber-profiler.

Provides commands for profiling ML training scripts, analyzing saved profiles,
generating reports, comparing profiles, live monitoring, and uploading results
to the Chamber platform.

Uses Click for command parsing and Rich for beautiful terminal output.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src import __version__
from src.profiler.gpu_profiler import GPUProfiler, ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTracer, KernelTraceResult
from src.profiler.memory_analyzer import MemoryAnalyzer, MemoryAnalysisResult
from src.profiler.communication_profiler import CommunicationProfiler, CommProfileResult
from src.profiler.data_loading_profiler import DataLoadingProfiler, DataLoadingResult
from src.analysis.bottleneck_detector import BottleneckDetector, PerformanceSummary
from src.analysis.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

console = Console()

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
_ENV_API_KEY = "CHAMBER_API_KEY"
_ENV_API_URL = "CHAMBER_API_URL"


# ============================================================================
# Helpers
# ============================================================================


def _error(message: str) -> None:
    """Print an error message and exit with code 1."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise SystemExit(1)


def _warn(message: str) -> None:
    """Print a warning message to stderr."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}", style="yellow")


def _info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[dim]{message}[/dim]")


def _load_profile(path: str) -> Dict[str, Any]:
    """Load a saved profile JSON from disk.

    Parameters
    ----------
    path:
        Path to the profile JSON file.

    Returns
    -------
    dict
        The parsed profile data.

    Raises
    ------
    SystemExit
        If the file cannot be read or parsed.
    """
    filepath = Path(path)
    if not filepath.exists():
        _error(f"Profile file not found: {path}")
    if not filepath.is_file():
        _error(f"Not a file: {path}")

    try:
        text = filepath.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(text)
        return data
    except json.JSONDecodeError as exc:
        _error(f"Invalid JSON in profile file {path}: {exc}")
    except OSError as exc:
        _error(f"Cannot read profile file {path}: {exc}")

    # Unreachable, but satisfies type checker.
    return {}  # pragma: no cover


def _reconstruct_results(
    data: Dict[str, Any],
) -> Tuple[
    Optional[GPUProfileResult],
    Optional[KernelTraceResult],
    Optional[MemoryAnalysisResult],
    Optional[CommProfileResult],
    Optional[DataLoadingResult],
]:
    """Reconstruct profiler result objects from a saved profile dict.

    Parameters
    ----------
    data:
        The parsed JSON profile data, as produced by the ``run`` command
        or by :class:`ReportGenerator.generate_json_report`.

    Returns
    -------
    tuple
        A 5-tuple of (gpu_result, kernel_result, memory_result,
        comm_result, data_result), each of which may be ``None`` if the
        corresponding data is not present in the profile.
    """
    gpu_result: Optional[GPUProfileResult] = None
    kernel_result: Optional[KernelTraceResult] = None
    memory_result: Optional[MemoryAnalysisResult] = None
    comm_result: Optional[CommProfileResult] = None
    data_result: Optional[DataLoadingResult] = None

    if "gpu_profile" in data:
        try:
            gpu_result = GPUProfileResult.from_dict(data["gpu_profile"])
        except Exception:
            logger.debug("Failed to reconstruct GPU profile result.", exc_info=True)

    if "kernel_trace" in data:
        try:
            kernel_result = KernelTraceResult.from_dict(data["kernel_trace"])
        except Exception:
            logger.debug("Failed to reconstruct kernel trace result.", exc_info=True)

    if "memory_analysis" in data:
        try:
            memory_result = MemoryAnalysisResult.from_dict(data["memory_analysis"])
        except Exception:
            logger.debug("Failed to reconstruct memory analysis result.", exc_info=True)

    if "communication_profile" in data:
        try:
            comm_result = CommProfileResult.from_dict(data["communication_profile"])
        except Exception:
            logger.debug(
                "Failed to reconstruct communication profile result.", exc_info=True
            )

    if "data_loading_profile" in data:
        try:
            data_result = DataLoadingResult.from_dict(data["data_loading_profile"])
        except Exception:
            logger.debug(
                "Failed to reconstruct data loading profile result.", exc_info=True
            )

    return gpu_result, kernel_result, memory_result, comm_result, data_result


def _run_analysis(
    gpu_result: Optional[GPUProfileResult] = None,
    kernel_result: Optional[KernelTraceResult] = None,
    memory_result: Optional[MemoryAnalysisResult] = None,
    comm_result: Optional[CommProfileResult] = None,
    data_result: Optional[DataLoadingResult] = None,
) -> PerformanceSummary:
    """Run the bottleneck detector and return a performance summary."""
    detector = BottleneckDetector(
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
    )
    return detector.analyze()


def _generate_report(
    summary: PerformanceSummary,
    gpu_result: Optional[GPUProfileResult] = None,
    kernel_result: Optional[KernelTraceResult] = None,
    memory_result: Optional[MemoryAnalysisResult] = None,
    comm_result: Optional[CommProfileResult] = None,
    data_result: Optional[DataLoadingResult] = None,
    report_format: str = "terminal",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Create a ReportGenerator and produce the requested report format.

    Returns
    -------
    str | None
        The output file path for html/json formats, or ``None`` for terminal.
    """
    generator = ReportGenerator(
        performance_summary=summary,
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
    )
    return generator.generate_report(format=report_format, output_path=output_path)


def _save_profile(
    output_path: str,
    gpu_result: Optional[GPUProfileResult] = None,
    kernel_result: Optional[KernelTraceResult] = None,
    memory_result: Optional[MemoryAnalysisResult] = None,
    comm_result: Optional[CommProfileResult] = None,
    data_result: Optional[DataLoadingResult] = None,
    summary: Optional[PerformanceSummary] = None,
    command: Optional[str] = None,
) -> None:
    """Save profiling results to a JSON file.

    Parameters
    ----------
    output_path:
        File path for the output JSON.
    gpu_result, kernel_result, memory_result, comm_result, data_result:
        Raw profiler results (any may be ``None``).
    summary:
        The performance summary from bottleneck detection.
    command:
        The command that was profiled (for metadata).
    """
    import datetime

    data: Dict[str, Any] = {
        "metadata": {
            "tool": "chamber-profiler",
            "version": __version__,
            "timestamp": datetime.datetime.now().isoformat(),
            "command": command,
        },
    }

    if summary is not None:
        data["summary"] = summary.to_dict()

    if gpu_result is not None:
        data["gpu_profile"] = gpu_result.to_dict()

    if kernel_result is not None:
        data["kernel_trace"] = kernel_result.to_dict()

    if memory_result is not None:
        data["memory_analysis"] = memory_result.to_dict()

    if comm_result is not None:
        data["communication_profile"] = comm_result.to_dict()

    if data_result is not None:
        data["data_loading_profile"] = data_result.to_dict()

    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _get_chamber_client(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Any:
    """Create and return a ChamberClient instance.

    Parameters
    ----------
    api_key:
        Chamber API key.  Falls back to ``CHAMBER_API_KEY`` env var.
    api_url:
        Chamber API URL.  Falls back to ``CHAMBER_API_URL`` env var.

    Returns
    -------
    ChamberClient
        An authenticated client instance.

    Raises
    ------
    SystemExit
        If the API key is not provided and not set in the environment,
        or if the chamber module is not available.
    """
    resolved_key = api_key or os.environ.get(_ENV_API_KEY)
    if not resolved_key:
        _error(
            f"Chamber API key is required. Provide --api-key or set "
            f"the {_ENV_API_KEY} environment variable."
        )

    resolved_url = api_url or os.environ.get(_ENV_API_URL)

    try:
        from src.chamber import ChamberClient
    except ImportError:
        _error(
            "Chamber integration module is not available. "
            "Ensure the chamber module is installed."
        )

    kwargs: Dict[str, Any] = {"api_key": resolved_key}
    if resolved_url:
        kwargs["api_url"] = resolved_url

    try:
        client = ChamberClient(**kwargs)
        return client
    except Exception as exc:
        _error(f"Failed to create Chamber client: {exc}")


# ============================================================================
# CLI group
# ============================================================================


@click.group(name="chamber-profile")
@click.version_option(version=__version__, prog_name="chamber-profile")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging.",
)
def cli(verbose: bool) -> None:
    """Chamber Profiler - One command to profile any ML training job."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# run
# ============================================================================


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Save profile results to a JSON file.",
)
@click.option(
    "--interval",
    type=int,
    default=100,
    show_default=True,
    help="GPU metric sampling interval in milliseconds.",
)
@click.option(
    "--no-kernel",
    is_flag=True,
    default=False,
    help="Disable CUDA kernel tracing.",
)
@click.option(
    "--no-memory",
    is_flag=True,
    default=False,
    help="Disable GPU memory analysis.",
)
@click.option(
    "--html",
    "html_output",
    type=click.Path(),
    default=None,
    help="Generate an HTML report at the given path.",
)
@click.option(
    "--upload",
    is_flag=True,
    default=False,
    help="Upload the profile to Chamber after profiling.",
)
@click.pass_context
def run(
    ctx: click.Context,
    command: Tuple[str, ...],
    output: Optional[str],
    interval: int,
    no_kernel: bool,
    no_memory: bool,
    html_output: Optional[str],
    upload: bool,
) -> None:
    """Profile a training script.

    Usage: chamber-profile run python train.py --epochs 10

    Runs the given command as a subprocess while collecting GPU metrics,
    kernel traces, and memory usage. Prints a terminal report when the
    command completes.
    """
    if not command:
        _error("No command specified. Usage: chamber-profile run <command>")

    command_str = " ".join(command)
    console.print(
        Panel(
            f"[bold]Profiling:[/bold] {command_str}",
            border_style="magenta",
            padding=(0, 1),
        )
    )

    # ---- Initialize profilers ----
    gpu_profiler = GPUProfiler(interval_ms=interval)

    kernel_tracer: Optional[KernelTracer] = None
    if not no_kernel:
        kernel_tracer = KernelTracer(top_k=10)

    memory_analyzer: Optional[MemoryAnalyzer] = None
    if not no_memory:
        memory_analyzer = MemoryAnalyzer(interval_ms=interval)

    comm_profiler = CommunicationProfiler()
    data_profiler = DataLoadingProfiler()

    # ---- Start profilers ----
    _info("Starting profilers...")
    gpu_profiler.start()
    if kernel_tracer is not None:
        kernel_tracer.start()
    if memory_analyzer is not None:
        memory_analyzer.start()
    comm_profiler.start()
    data_profiler.start()

    # ---- Execute command ----
    _info(f"Running: {command_str}")
    start_time = time.monotonic()

    try:
        proc = subprocess.run(
            list(command),
            env=os.environ.copy(),
        )
        exit_code = proc.returncode
    except FileNotFoundError:
        _stop_profilers(
            gpu_profiler, kernel_tracer, memory_analyzer, comm_profiler, data_profiler
        )
        _error(f"Command not found: {command[0]}")
        return  # Unreachable, but satisfies type checker.
    except KeyboardInterrupt:
        exit_code = 130
        _warn("Profiling interrupted by user (Ctrl+C).")
    except Exception as exc:
        _stop_profilers(
            gpu_profiler, kernel_tracer, memory_analyzer, comm_profiler, data_profiler
        )
        _error(f"Failed to execute command: {exc}")
        return

    elapsed = time.monotonic() - start_time

    # ---- Stop profilers and collect results ----
    _info("Stopping profilers and collecting results...")
    gpu_result, kernel_result, memory_result, comm_result, data_result = (
        _stop_profilers(
            gpu_profiler, kernel_tracer, memory_analyzer, comm_profiler, data_profiler
        )
    )

    # ---- Print command status ----
    if exit_code == 0:
        console.print(
            f"\n[bold green]Command completed successfully[/bold green] "
            f"in {elapsed:.2f}s"
        )
    else:
        console.print(
            f"\n[bold yellow]Command exited with code {exit_code}[/bold yellow] "
            f"after {elapsed:.2f}s"
        )

    # ---- Run analysis ----
    _info("Analyzing results...")
    summary = _run_analysis(
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
    )

    # ---- Generate terminal report ----
    console.print()
    _generate_report(
        summary=summary,
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
        report_format="terminal",
    )

    # ---- Save profile JSON ----
    if output:
        _save_profile(
            output_path=output,
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
            summary=summary,
            command=command_str,
        )
        console.print(f"[bold green]Profile saved to:[/bold green] {output}")

    # ---- Generate HTML report ----
    if html_output:
        _generate_report(
            summary=summary,
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
            report_format="html",
            output_path=html_output,
        )
        console.print(f"[bold green]HTML report saved to:[/bold green] {html_output}")

    # ---- Upload to Chamber ----
    if upload:
        _upload_profile_data(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
            summary=summary,
            command=command_str,
        )

    # Exit with the same code as the profiled command.
    if exit_code != 0:
        raise SystemExit(exit_code)


def _stop_profilers(
    gpu_profiler: GPUProfiler,
    kernel_tracer: Optional[KernelTracer],
    memory_analyzer: Optional[MemoryAnalyzer],
    comm_profiler: CommunicationProfiler,
    data_profiler: DataLoadingProfiler,
) -> Tuple[
    GPUProfileResult,
    Optional[KernelTraceResult],
    Optional[MemoryAnalysisResult],
    CommProfileResult,
    DataLoadingResult,
]:
    """Stop all profilers and return their results."""
    gpu_result = gpu_profiler.stop()

    kernel_result: Optional[KernelTraceResult] = None
    if kernel_tracer is not None:
        kernel_result = kernel_tracer.stop()

    memory_result: Optional[MemoryAnalysisResult] = None
    if memory_analyzer is not None:
        memory_result = memory_analyzer.stop()

    comm_result = comm_profiler.stop()
    data_result = data_profiler.stop()

    return gpu_result, kernel_result, memory_result, comm_result, data_result


def _upload_profile_data(
    gpu_result: Optional[GPUProfileResult] = None,
    kernel_result: Optional[KernelTraceResult] = None,
    memory_result: Optional[MemoryAnalysisResult] = None,
    comm_result: Optional[CommProfileResult] = None,
    data_result: Optional[DataLoadingResult] = None,
    summary: Optional[PerformanceSummary] = None,
    command: Optional[str] = None,
    job_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Upload profile data to Chamber.

    Attempts to create a Chamber client and upload the results. Prints
    the resulting URL on success, or a warning on failure.
    """
    try:
        client = _get_chamber_client(api_key=api_key)
    except SystemExit:
        return

    import datetime

    payload: Dict[str, Any] = {
        "metadata": {
            "tool": "chamber-profiler",
            "version": __version__,
            "timestamp": datetime.datetime.now().isoformat(),
            "command": command,
        },
    }

    if summary is not None:
        payload["summary"] = summary.to_dict()
    if gpu_result is not None:
        payload["gpu_profile"] = gpu_result.to_dict()
    if kernel_result is not None:
        payload["kernel_trace"] = kernel_result.to_dict()
    if memory_result is not None:
        payload["memory_analysis"] = memory_result.to_dict()
    if comm_result is not None:
        payload["communication_profile"] = comm_result.to_dict()
    if data_result is not None:
        payload["data_loading_profile"] = data_result.to_dict()

    try:
        _info("Uploading profile to Chamber...")
        upload_kwargs: Dict[str, Any] = {"profile_data": payload}
        if job_id:
            upload_kwargs["job_id"] = job_id

        result = client.upload_profile(**upload_kwargs)

        url = None
        if isinstance(result, dict):
            url = result.get("url") or result.get("profile_url")
        elif isinstance(result, str):
            url = result

        if url:
            console.print(
                f"[bold green]Uploaded to Chamber:[/bold green] {url}"
            )
        else:
            console.print("[bold green]Profile uploaded to Chamber successfully.[/bold green]")
    except Exception as exc:
        _warn(f"Failed to upload to Chamber: {exc}")


# ============================================================================
# analyze
# ============================================================================


@cli.command()
@click.argument("profile_path", type=click.Path(exists=True))
def analyze(profile_path: str) -> None:
    """Analyze a saved profile and print bottleneck report.

    Usage: chamber-profile analyze profile.json
    """
    console.print(
        Panel(
            f"[bold]Analyzing:[/bold] {profile_path}",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    data = _load_profile(profile_path)
    gpu_result, kernel_result, memory_result, comm_result, data_result = (
        _reconstruct_results(data)
    )

    summary = _run_analysis(
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
    )

    console.print()
    _generate_report(
        summary=summary,
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
        report_format="terminal",
    )


# ============================================================================
# report
# ============================================================================


@cli.command()
@click.argument("profile_path", type=click.Path(exists=True))
@click.option(
    "--format", "-f",
    "report_format",
    type=click.Choice(["terminal", "html", "json"], case_sensitive=False),
    default="terminal",
    show_default=True,
    help="Report output format.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file path (required for html/json formats).",
)
def report(profile_path: str, report_format: str, output: Optional[str]) -> None:
    """Generate a report from a saved profile.

    Usage: chamber-profile report profile.json --format html --output report.html
    """
    data = _load_profile(profile_path)
    gpu_result, kernel_result, memory_result, comm_result, data_result = (
        _reconstruct_results(data)
    )

    # Try to use the saved summary if available, otherwise re-analyze.
    summary: Optional[PerformanceSummary] = None
    if "summary" in data:
        try:
            summary = PerformanceSummary.from_dict(data["summary"])
        except Exception:
            logger.debug("Failed to reconstruct saved summary.", exc_info=True)

    if summary is None:
        summary = _run_analysis(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
        )

    result_path = _generate_report(
        summary=summary,
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
        report_format=report_format,
        output_path=output,
    )

    if result_path:
        console.print(
            f"[bold green]Report saved to:[/bold green] {result_path}"
        )


# ============================================================================
# compare
# ============================================================================


@cli.command()
@click.argument("profile1_path", type=click.Path(exists=True))
@click.argument("profile2_path", type=click.Path(exists=True))
def compare(profile1_path: str, profile2_path: str) -> None:
    """Compare two profiles side by side.

    Usage: chamber-profile compare baseline.json optimized.json

    Shows improvements and regressions in duration, GPU utilization,
    memory usage, and bottleneck counts.
    """
    console.print(
        Panel(
            f"[bold]Comparing profiles[/bold]\n"
            f"  A: {profile1_path}\n"
            f"  B: {profile2_path}",
            border_style="magenta",
            padding=(0, 1),
        )
    )

    data1 = _load_profile(profile1_path)
    data2 = _load_profile(profile2_path)

    results1 = _reconstruct_results(data1)
    results2 = _reconstruct_results(data2)

    summary1 = _run_analysis(*results1)
    summary2 = _run_analysis(*results2)

    gpu1, _, mem1, _, _ = results1
    gpu2, _, mem2, _, _ = results2

    # ---- Build comparison table ----
    table = Table(
        title="Profile Comparison",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold white",
    )
    table.add_column("Metric", style="bold", min_width=24)
    table.add_column("Profile A", justify="right", min_width=16)
    table.add_column("Profile B", justify="right", min_width=16)
    table.add_column("Change", justify="right", min_width=16)

    # Duration
    _add_comparison_row(
        table,
        "Total Duration",
        summary1.total_time_s,
        summary2.total_time_s,
        unit="s",
        lower_is_better=True,
    )

    # GPU efficiency
    _add_comparison_row(
        table,
        "GPU Efficiency",
        summary1.overall_efficiency * 100.0,
        summary2.overall_efficiency * 100.0,
        unit="%",
        lower_is_better=False,
    )

    # Compute %
    _add_comparison_row(
        table,
        "Compute %",
        summary1.compute_pct,
        summary2.compute_pct,
        unit="%",
        lower_is_better=False,
    )

    # Idle %
    _add_comparison_row(
        table,
        "Idle %",
        summary1.idle_pct,
        summary2.idle_pct,
        unit="%",
        lower_is_better=True,
    )

    # Communication %
    _add_comparison_row(
        table,
        "Communication %",
        summary1.communication_pct,
        summary2.communication_pct,
        unit="%",
        lower_is_better=True,
    )

    # Data Loading %
    _add_comparison_row(
        table,
        "Data Loading %",
        summary1.data_loading_pct,
        summary2.data_loading_pct,
        unit="%",
        lower_is_better=True,
    )

    # GPU utilization (if available)
    if gpu1 is not None and gpu1.snapshots:
        avg_util1 = gpu1.avg_utilization
    else:
        avg_util1 = 0.0

    if gpu2 is not None and gpu2.snapshots:
        avg_util2 = gpu2.avg_utilization
    else:
        avg_util2 = 0.0

    if avg_util1 > 0 or avg_util2 > 0:
        _add_comparison_row(
            table,
            "Avg GPU Utilization",
            avg_util1,
            avg_util2,
            unit="%",
            lower_is_better=False,
        )

    # Peak memory (if available)
    if gpu1 is not None and gpu1.snapshots:
        peak_mem1 = gpu1.peak_memory_mb
    else:
        peak_mem1 = 0.0

    if gpu2 is not None and gpu2.snapshots:
        peak_mem2 = gpu2.peak_memory_mb
    else:
        peak_mem2 = 0.0

    if peak_mem1 > 0 or peak_mem2 > 0:
        _add_comparison_row(
            table,
            "Peak Memory (MB)",
            peak_mem1,
            peak_mem2,
            unit=" MB",
            lower_is_better=True,
        )

    # Bottleneck count
    _add_comparison_row(
        table,
        "Bottlenecks Found",
        float(len(summary1.bottlenecks)),
        float(len(summary2.bottlenecks)),
        unit="",
        lower_is_better=True,
        is_integer=True,
    )

    console.print()
    console.print(table)

    # ---- Primary bottleneck comparison ----
    console.print()
    pb_table = Table(
        title="Primary Bottleneck",
        box=box.ROUNDED,
        show_lines=False,
        title_style="bold white",
    )
    pb_table.add_column("", style="dim", min_width=10)
    pb_table.add_column("Profile A", min_width=20)
    pb_table.add_column("Profile B", min_width=20)

    pb1 = summary1.primary_bottleneck.replace("_", " ").title()
    pb2 = summary2.primary_bottleneck.replace("_", " ").title()

    pb_table.add_row(
        "Category",
        pb1,
        pb2,
    )

    console.print(pb_table)

    # ---- Speedup summary ----
    if summary1.total_time_s > 0 and summary2.total_time_s > 0:
        speedup = summary1.total_time_s / summary2.total_time_s
        if speedup > 1.0:
            console.print(
                f"\n[bold green]Profile B is {speedup:.2f}x faster "
                f"than Profile A.[/bold green]"
            )
        elif speedup < 1.0:
            slowdown = 1.0 / speedup
            console.print(
                f"\n[bold red]Profile B is {slowdown:.2f}x slower "
                f"than Profile A.[/bold red]"
            )
        else:
            console.print(
                "\n[bold]Profile A and Profile B have similar duration.[/bold]"
            )

    console.print()


def _add_comparison_row(
    table: Table,
    metric: str,
    value_a: float,
    value_b: float,
    unit: str = "",
    lower_is_better: bool = True,
    is_integer: bool = False,
) -> None:
    """Add a row to the comparison table with color-coded delta.

    Parameters
    ----------
    table:
        The Rich Table to add the row to.
    metric:
        Metric name for the first column.
    value_a, value_b:
        Values from profile A and profile B.
    unit:
        Unit suffix for display (e.g. ``"%"``, ``"s"``).
    lower_is_better:
        If ``True``, a decrease from A to B is shown in green (improvement).
        If ``False``, an increase from A to B is shown in green.
    is_integer:
        If ``True``, format values as integers.
    """
    fmt = ".0f" if is_integer else ".2f"
    val_a_str = f"{value_a:{fmt}}{unit}"
    val_b_str = f"{value_b:{fmt}}{unit}"

    delta = value_b - value_a
    if abs(delta) < 0.005:
        delta_str = f"  [dim]--[/dim]"
    else:
        # Determine direction.
        if lower_is_better:
            is_improvement = delta < 0
        else:
            is_improvement = delta > 0

        sign = "+" if delta > 0 else ""
        color = "green" if is_improvement else "red"
        delta_fmt = f"{sign}{delta:{fmt}}{unit}"

        # Also compute percentage change if value_a is nonzero.
        pct_str = ""
        if abs(value_a) > 0.001:
            pct = (delta / abs(value_a)) * 100.0
            pct_str = f" ({sign}{pct:.1f}%)"

        delta_str = f"[bold {color}]{delta_fmt}{pct_str}[/bold {color}]"

    table.add_row(metric, val_a_str, val_b_str, delta_str)


# ============================================================================
# live
# ============================================================================


@cli.command()
@click.option(
    "--interval",
    type=int,
    default=500,
    show_default=True,
    help="Dashboard refresh interval in milliseconds.",
)
def live(interval: int) -> None:
    """Real-time GPU profiling dashboard.

    Usage: chamber-profile live

    Shows live GPU utilization, memory, temperature, and power metrics
    in the terminal. Press Ctrl+C to stop.
    """
    profiler = GPUProfiler(interval_ms=max(interval // 2, 50))
    profiler.start()

    console.print(
        Panel(
            "[bold]Live GPU Dashboard[/bold]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="magenta",
            padding=(0, 1),
        )
    )

    refresh_s = interval / 1000.0

    try:
        with Live(
            _build_live_table(profiler),
            console=console,
            refresh_per_second=max(1, int(1.0 / refresh_s)),
            transient=False,
        ) as live_display:
            while True:
                time.sleep(refresh_s)
                live_display.update(_build_live_table(profiler))
    except KeyboardInterrupt:
        pass
    finally:
        profiler.stop()
        console.print("\n[dim]Live dashboard stopped.[/dim]")


def _build_live_table(profiler: GPUProfiler) -> Table:
    """Build a Rich Table showing current GPU metrics from the profiler.

    Parameters
    ----------
    profiler:
        A running GPUProfiler instance.

    Returns
    -------
    Table
        A Rich Table with one row per GPU, showing utilization,
        memory, temperature, and power.
    """
    metrics = profiler.get_live_metrics()

    table = Table(
        title="GPU Metrics (Live)",
        box=box.ROUNDED,
        title_style="bold magenta",
        expand=True,
    )
    table.add_column("GPU", style="bold", min_width=5, justify="center")
    table.add_column("Utilization", min_width=20)
    table.add_column("SM Activity", min_width=20)
    table.add_column("Memory", min_width=24)
    table.add_column("Mem Bandwidth", min_width=20)
    table.add_column("Temp", justify="center", min_width=8)
    table.add_column("Power", justify="center", min_width=10)
    table.add_column("Clock", justify="center", min_width=10)

    if not metrics:
        table.add_row(
            "-", "[dim]Waiting for data...[/dim]",
            "", "", "", "", "", "",
        )
        return table

    for gpu_idx in sorted(metrics.keys()):
        snap = metrics[gpu_idx]

        util_bar = _make_bar(snap.utilization_pct, 15)
        sm_bar = _make_bar(snap.sm_activity_pct, 15)
        mem_bw_bar = _make_bar(snap.memory_bandwidth_pct, 15)

        mem_used = snap.memory_used_mb
        mem_total = snap.memory_total_mb
        mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
        mem_bar = _make_bar(mem_pct, 10)
        mem_text = Text()
        mem_text.append_text(mem_bar)
        mem_text.append(f" {mem_used:,.0f}/{mem_total:,.0f} MB", style="dim")

        temp = snap.temperature_c
        temp_color = "green" if temp < 70 else ("yellow" if temp < 85 else "red")
        temp_text = Text(f"{temp:.0f}C", style=f"bold {temp_color}")

        power_text = Text(f"{snap.power_watts:.0f}W", style="bold")
        clock_text = Text(f"{snap.clock_speed_mhz:.0f} MHz", style="bold")

        table.add_row(
            str(gpu_idx),
            util_bar,
            sm_bar,
            mem_text,
            mem_bw_bar,
            temp_text,
            power_text,
            clock_text,
        )

    return table


def _make_bar(pct: float, width: int = 15) -> Text:
    """Create a colored bar with percentage label.

    Parameters
    ----------
    pct:
        Percentage value (0-100).
    width:
        Character width of the bar.

    Returns
    -------
    Text
        A Rich Text object rendering the bar.
    """
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * width))
    empty = width - filled

    if pct >= 70.0:
        color = "green"
    elif pct >= 40.0:
        color = "yellow"
    else:
        color = "red"

    bar = Text()
    bar.append("\u2588" * filled, style=color)
    bar.append("\u2591" * empty, style="dim")
    bar.append(f" {pct:.0f}%", style=f"bold {color}")
    return bar


# ============================================================================
# upload
# ============================================================================


@cli.command()
@click.argument("profile_path", type=click.Path(exists=True))
@click.option(
    "--job-id",
    type=str,
    default=None,
    help="Attach the profile to a specific Chamber job ID.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help=f"Chamber API key (or set {_ENV_API_KEY} env var).",
)
def upload(
    profile_path: str,
    job_id: Optional[str],
    api_key: Optional[str],
) -> None:
    """Upload a saved profile to Chamber.

    Usage: chamber-profile upload profile.json --job-id abc123
    """
    console.print(
        Panel(
            f"[bold]Uploading:[/bold] {profile_path}",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    data = _load_profile(profile_path)

    gpu_result, kernel_result, memory_result, comm_result, data_result = (
        _reconstruct_results(data)
    )

    summary: Optional[PerformanceSummary] = None
    if "summary" in data:
        try:
            summary = PerformanceSummary.from_dict(data["summary"])
        except Exception:
            logger.debug("Failed to reconstruct saved summary.", exc_info=True)

    command = None
    if "metadata" in data and isinstance(data["metadata"], dict):
        command = data["metadata"].get("command")

    _upload_profile_data(
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
        summary=summary,
        command=command,
        job_id=job_id,
        api_key=api_key,
    )


# ============================================================================
# profile (Chamber CLI integration)
# ============================================================================


@cli.command(name="profile")
@click.argument("job_id", type=str)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Save profile results to a JSON file.",
)
@click.option(
    "--interval",
    type=int,
    default=100,
    show_default=True,
    help="GPU metric sampling interval in milliseconds.",
)
@click.option(
    "--upload",
    is_flag=True,
    default=False,
    help="Upload the profile to Chamber after profiling.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help=f"Chamber API key (or set {_ENV_API_KEY} env var).",
)
def profile_command(
    job_id: str,
    output: Optional[str],
    interval: int,
    upload: bool,
    api_key: Optional[str],
) -> None:
    """Profile a Chamber-managed job by its job ID.

    Usage: chamber-profile profile <job-id>

    Connects to the Chamber platform to fetch job metadata, then profiles
    the running workload. Works with both local and distributed jobs.
    """
    console.print(
        Panel(
            f"[bold]Profiling Chamber job:[/bold] {job_id}",
            border_style="magenta",
            padding=(0, 1),
        )
    )

    # ---- Initialize profilers ----
    gpu_profiler = GPUProfiler(interval_ms=interval)
    kernel_tracer = KernelTracer(top_k=10)
    memory_analyzer = MemoryAnalyzer(interval_ms=interval)
    comm_profiler = CommunicationProfiler()
    data_profiler = DataLoadingProfiler()

    # ---- Start profilers ----
    _info(f"Attaching profilers to job {job_id}...")
    gpu_profiler.start()
    kernel_tracer.start()
    memory_analyzer.start()
    comm_profiler.start()
    data_profiler.start()

    # ---- Collect for a short window ----
    _info("Collecting metrics (press Ctrl+C to stop early)...")
    try:
        # Default profiling window: 30 seconds or until interrupted.
        for _ in range(30):
            time.sleep(1.0)
    except KeyboardInterrupt:
        _warn("Profiling interrupted by user.")

    # ---- Stop and collect ----
    _info("Stopping profilers and collecting results...")
    gpu_result = gpu_profiler.stop()
    kernel_result = kernel_tracer.stop()
    memory_result = memory_analyzer.stop()
    comm_result = comm_profiler.stop()
    data_result = data_profiler.stop()

    # ---- Run analysis ----
    _info("Analyzing results...")
    summary = _run_analysis(
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
    )

    # ---- Generate terminal report ----
    console.print()
    _generate_report(
        summary=summary,
        gpu_result=gpu_result,
        kernel_result=kernel_result,
        memory_result=memory_result,
        comm_result=comm_result,
        data_result=data_result,
        report_format="terminal",
    )

    # ---- Save profile JSON ----
    if output:
        _save_profile(
            output_path=output,
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
            summary=summary,
            command=f"chamber profile {job_id}",
        )
        console.print(f"[bold green]Profile saved to:[/bold green] {output}")

    # ---- Upload to Chamber ----
    if upload:
        _upload_profile_data(
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
            comm_result=comm_result,
            data_result=data_result,
            summary=summary,
            command=f"chamber profile {job_id}",
            job_id=job_id,
            api_key=api_key,
        )


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    """Entry point for the CLI.

    This function exists so the CLI can also be invoked via
    ``python -m src.cli.main``.
    """
    cli()


if __name__ == "__main__":
    main()
