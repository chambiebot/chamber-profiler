"""Report generation module for chamber-profiler.

Generates rich terminal reports, self-contained HTML reports, and structured
JSON exports from profiling and bottleneck analysis results.

Terminal output uses the Rich library for beautiful console rendering with
tables, panels, progress bars, and color-coded metrics.  HTML reports are
fully self-contained (inline CSS and JS, no external dependencies).  JSON
exports include all raw data alongside the analysis summary for downstream
tooling.
"""

from __future__ import annotations

import datetime
import html
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.bottleneck_detector import (
    Bottleneck,
    PerformanceSummary,
    TimeCategory,
)
from src.profiler.gpu_profiler import ProfileResult as GPUProfileResult
from src.profiler.kernel_tracer import KernelTraceResult
from src.profiler.memory_analyzer import MemoryAnalysisResult
from src.profiler.communication_profiler import CommProfileResult
from src.profiler.data_loading_profiler import DataLoadingResult

logger = logging.getLogger(__name__)


# ============================================================================
# Color thresholds
# ============================================================================

# Utilization / efficiency thresholds for color coding.
_GOOD_THRESHOLD: float = 70.0   # >= 70% = green
_OK_THRESHOLD: float = 40.0     # >= 40% = yellow
# < 40% = red


def _efficiency_color(value: float) -> str:
    """Return a Rich color name for the given utilization/efficiency percentage."""
    if value >= _GOOD_THRESHOLD:
        return "green"
    if value >= _OK_THRESHOLD:
        return "yellow"
    return "red"


def _efficiency_hex(value: float) -> str:
    """Return a CSS hex color for the given utilization/efficiency percentage."""
    if value >= _GOOD_THRESHOLD:
        return "#4ade80"
    if value >= _OK_THRESHOLD:
        return "#facc15"
    return "#f87171"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _truncate_kernel_name(name: str, max_len: int = 60) -> str:
    """Truncate a CUDA kernel name for display, preserving the tail."""
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3):]


# ============================================================================
# Report Generator
# ============================================================================


class ReportGenerator:
    """Generate profiling reports in multiple formats.

    Accepts a :class:`PerformanceSummary` from the bottleneck detector and
    optional raw results from each profiler.  Produces terminal, HTML, and
    JSON reports.

    Usage::

        generator = ReportGenerator(
            performance_summary=summary,
            gpu_result=gpu_result,
            kernel_result=kernel_result,
            memory_result=memory_result,
        )
        generator.generate_report(format="terminal")
        generator.generate_report(format="html", output_path="report.html")
        generator.generate_report(format="json", output_path="report.json")
    """

    def __init__(
        self,
        performance_summary: PerformanceSummary,
        gpu_result: Optional[GPUProfileResult] = None,
        kernel_result: Optional[KernelTraceResult] = None,
        memory_result: Optional[MemoryAnalysisResult] = None,
        comm_result: Optional[CommProfileResult] = None,
        data_result: Optional[DataLoadingResult] = None,
    ) -> None:
        self._summary = performance_summary
        self._gpu_result = gpu_result
        self._kernel_result = kernel_result
        self._memory_result = memory_result
        self._comm_result = comm_result
        self._data_result = data_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        format: str = "terminal",
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Dispatch to the appropriate report generation method.

        Parameters
        ----------
        format:
            One of ``"terminal"``, ``"html"``, or ``"json"``.
        output_path:
            File path for HTML and JSON output.  Ignored for terminal format.

        Returns
        -------
        str | None
            The output file path for HTML/JSON, or ``None`` for terminal.

        Raises
        ------
        ValueError
            If *format* is not recognised.
        """
        fmt = format.lower().strip()
        if fmt == "terminal":
            self.generate_terminal_report()
            return None
        elif fmt == "html":
            if output_path is None:
                output_path = "chamber_profile_report.html"
            self.generate_html_report(output_path)
            return output_path
        elif fmt == "json":
            if output_path is None:
                output_path = "chamber_profile_report.json"
            self.generate_json_report(output_path)
            return output_path
        else:
            raise ValueError(
                f"Unknown report format {fmt!r}. "
                f"Expected one of: 'terminal', 'html', 'json'."
            )

    # ==================================================================
    # Terminal report
    # ==================================================================

    def generate_terminal_report(self) -> None:
        """Print a rich terminal report using the Rich library.

        The report includes a header, efficiency score, time breakdown table,
        bottleneck analysis, GPU metrics, top kernels, memory breakdown,
        and a summary with the primary recommendation.
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich import box
        except ImportError:
            logger.error(
                "The 'rich' library is required for terminal reports. "
                "Install it with: pip install rich"
            )
            # Fall back to plain text summary.
            print(self._summary.summary_text)
            return

        console = Console()
        summary = self._summary

        # -- Header --------------------------------------------------------
        header_text = Text()
        header_text.append("Chamber Profiler", style="bold magenta")
        header_text.append(" - Performance Analysis Report", style="bold white")
        console.print()
        console.print(
            Panel(
                header_text,
                border_style="magenta",
                padding=(1, 2),
            )
        )

        # -- Job info ------------------------------------------------------
        info_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            expand=False,
        )
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        info_table.add_row("Total profiled time", _format_duration(summary.total_time_s))
        info_table.add_row("Primary bottleneck", summary.primary_bottleneck.replace("_", " ").title())
        info_table.add_row("Report generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self._gpu_result is not None:
            info_table.add_row("GPUs profiled", str(self._gpu_result.gpu_count))

        console.print(info_table)
        console.print()

        # -- Overall efficiency score --------------------------------------
        efficiency_pct = summary.overall_efficiency * 100.0
        color = _efficiency_color(efficiency_pct)
        bar_width = 40
        filled = int(round(efficiency_pct / 100.0 * bar_width))
        empty = bar_width - filled
        bar = Text()
        bar.append("[", style="dim")
        bar.append("\u2588" * filled, style=color)
        bar.append("\u2591" * empty, style="dim")
        bar.append("]", style="dim")
        bar.append(f" {efficiency_pct:.1f}%", style=f"bold {color}")

        console.print(
            Panel(
                bar,
                title="[bold]Overall GPU Efficiency[/bold]",
                border_style=color,
                padding=(0, 1),
            )
        )
        console.print()

        # -- Time breakdown table ------------------------------------------
        self._print_time_breakdown_table(console, summary)
        console.print()

        # -- Top bottlenecks with recommendations --------------------------
        if summary.bottlenecks:
            self._print_bottlenecks_table(console, summary.bottlenecks)
            console.print()

        # -- GPU metrics summary -------------------------------------------
        if self._gpu_result is not None and self._gpu_result.snapshots:
            self._print_gpu_metrics_table(console)
            console.print()

        # -- Top-10 slowest kernels ----------------------------------------
        if self._kernel_result is not None and self._kernel_result.top_kernels:
            self._print_top_kernels_table(console)
            console.print()

        # -- Memory breakdown panel ----------------------------------------
        if self._memory_result is not None:
            self._print_memory_panel(console)
            console.print()

        # -- Summary text --------------------------------------------------
        summary_text = Text()
        summary_text.append(summary.summary_text)
        console.print(
            Panel(
                summary_text,
                title="[bold]Summary[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()

    def _print_time_breakdown_table(
        self,
        console: Any,
        summary: PerformanceSummary,
    ) -> None:
        """Print the time breakdown table to the Rich console."""
        from rich.table import Table
        from rich.text import Text
        from rich import box

        table = Table(
            title="Time Breakdown",
            box=box.ROUNDED,
            show_lines=False,
            title_style="bold white",
        )
        table.add_column("Category", style="bold", min_width=16)
        table.add_column("Percentage", justify="right", min_width=10)
        table.add_column("Bar", min_width=30)
        table.add_column("Time", justify="right", min_width=10)

        categories = [
            ("Compute", summary.compute_pct, "cyan"),
            ("Memory", summary.memory_pct, "blue"),
            ("Communication", summary.communication_pct, "magenta"),
            ("Data Loading", summary.data_loading_pct, "yellow"),
            ("Idle", summary.idle_pct, "red"),
        ]

        total_time = summary.total_time_s
        for name, pct, color in categories:
            bar_width = 25
            filled = int(round(pct / 100.0 * bar_width))
            empty = bar_width - filled
            bar = Text()
            bar.append("\u2588" * filled, style=color)
            bar.append("\u2591" * empty, style="dim")

            cat_time = total_time * (pct / 100.0)
            table.add_row(
                name,
                f"{pct:.1f}%",
                bar,
                _format_duration(cat_time),
            )

        console.print(table)

    def _print_bottlenecks_table(
        self,
        console: Any,
        bottlenecks: List[Bottleneck],
    ) -> None:
        """Print the bottlenecks and recommendations table."""
        from rich.table import Table
        from rich.text import Text
        from rich import box

        table = Table(
            title="Top Bottlenecks & Recommendations",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold white",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Category", style="bold", min_width=14)
        table.add_column("Impact", justify="center", min_width=8)
        table.add_column("Description", min_width=30, max_width=50)
        table.add_column("Top Recommendation", min_width=30, max_width=50)
        table.add_column("Speedup", justify="center", min_width=8)

        for i, b in enumerate(bottlenecks[:5], start=1):
            impact_pct = b.impact_score * 100.0
            impact_color = _efficiency_color(100.0 - impact_pct)
            impact_text = Text(f"{impact_pct:.0f}%", style=f"bold {impact_color}")

            category_label = b.category.value.replace("_", " ").title()

            top_rec = b.recommendations[0] if b.recommendations else "-"
            # Truncate long recommendations for the table.
            if len(top_rec) > 120:
                top_rec = top_rec[:117] + "..."

            table.add_row(
                str(i),
                category_label,
                impact_text,
                b.description,
                top_rec,
                b.expected_speedup or "-",
            )

        console.print(table)

    def _print_gpu_metrics_table(self, console: Any) -> None:
        """Print the GPU metrics summary table."""
        from rich.table import Table
        from rich.text import Text
        from rich import box

        assert self._gpu_result is not None
        snapshots = self._gpu_result.snapshots

        avg_util = sum(s.utilization_pct for s in snapshots) / len(snapshots)
        peak_mem = max(s.memory_used_mb for s in snapshots)
        total_mem = max(s.memory_total_mb for s in snapshots) if snapshots else 0.0
        avg_power = sum(s.power_watts for s in snapshots) / len(snapshots)
        avg_temp = sum(s.temperature_c for s in snapshots) / len(snapshots)
        avg_sm = sum(s.sm_activity_pct for s in snapshots) / len(snapshots)
        avg_mem_bw = sum(s.memory_bandwidth_pct for s in snapshots) / len(snapshots)
        avg_clock = sum(s.clock_speed_mhz for s in snapshots) / len(snapshots)

        table = Table(
            title="GPU Metrics Summary",
            box=box.ROUNDED,
            show_lines=False,
            title_style="bold white",
        )
        table.add_column("Metric", style="bold", min_width=24)
        table.add_column("Value", justify="right", min_width=14)

        util_color = _efficiency_color(avg_util)
        table.add_row(
            "Avg GPU Utilization",
            Text(f"{avg_util:.1f}%", style=f"bold {util_color}"),
        )

        sm_color = _efficiency_color(avg_sm)
        table.add_row(
            "Avg SM Activity",
            Text(f"{avg_sm:.1f}%", style=f"bold {sm_color}"),
        )

        table.add_row(
            "Avg Memory Bandwidth",
            Text(f"{avg_mem_bw:.1f}%", style="bold"),
        )

        mem_usage_pct = (peak_mem / total_mem * 100.0) if total_mem > 0 else 0.0
        mem_color = _efficiency_color(100.0 - mem_usage_pct)
        table.add_row(
            "Peak Memory Usage",
            Text(
                f"{peak_mem:.0f} / {total_mem:.0f} MB ({mem_usage_pct:.1f}%)",
                style=f"bold {mem_color}",
            ),
        )

        table.add_row(
            "Avg Power Draw",
            Text(f"{avg_power:.1f} W", style="bold"),
        )

        temp_color = "green" if avg_temp < 70 else ("yellow" if avg_temp < 85 else "red")
        table.add_row(
            "Avg Temperature",
            Text(f"{avg_temp:.1f} C", style=f"bold {temp_color}"),
        )

        table.add_row(
            "Avg Clock Speed",
            Text(f"{avg_clock:.0f} MHz", style="bold"),
        )

        table.add_row(
            "Profiling Duration",
            Text(_format_duration(self._gpu_result.duration_s), style="bold"),
        )

        console.print(table)

    def _print_top_kernels_table(self, console: Any) -> None:
        """Print the top-10 slowest kernels table."""
        from rich.table import Table
        from rich import box

        assert self._kernel_result is not None
        top_kernels = self._kernel_result.top_kernels[:10]
        total_us = self._kernel_result.total_kernel_time_us

        table = Table(
            title="Top-10 Slowest Kernels",
            box=box.ROUNDED,
            show_lines=False,
            title_style="bold white",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Kernel Name", min_width=30, max_width=55)
        table.add_column("Category", min_width=12)
        table.add_column("Duration (us)", justify="right", min_width=12)
        table.add_column("% of Total", justify="right", min_width=10)
        table.add_column("Occupancy", justify="right", min_width=10)

        for i, kernel in enumerate(top_kernels, start=1):
            pct = (kernel.duration_us / total_us * 100.0) if total_us > 0 else 0.0

            occ_str = "-"
            if kernel.occupancy is not None:
                occ_pct = kernel.occupancy * 100.0
                occ_color = _efficiency_color(occ_pct)
                occ_str = f"[{occ_color}]{occ_pct:.1f}%[/{occ_color}]"

            table.add_row(
                str(i),
                _truncate_kernel_name(kernel.name),
                kernel.category,
                f"{kernel.duration_us:,.1f}",
                f"{pct:.1f}%",
                occ_str,
            )

        console.print(table)

    def _print_memory_panel(self, console: Any) -> None:
        """Print a memory breakdown panel."""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box

        assert self._memory_result is not None

        lines: List[str] = []

        # Breakdown
        bd = self._memory_result.breakdown
        if bd is not None:
            total_known = (
                bd.parameter_memory_mb
                + bd.gradient_memory_mb
                + bd.optimizer_state_mb
                + bd.activation_memory_mb
                + bd.other_mb
            )

            table = Table(
                box=box.SIMPLE,
                show_header=True,
                padding=(0, 1),
            )
            table.add_column("Category", style="bold", min_width=20)
            table.add_column("Memory (MB)", justify="right", min_width=12)
            table.add_column("% of Total", justify="right", min_width=10)

            breakdown_items = [
                ("Parameters", bd.parameter_memory_mb),
                ("Gradients", bd.gradient_memory_mb),
                ("Optimizer State", bd.optimizer_state_mb),
                ("Activations", bd.activation_memory_mb),
                ("Other", bd.other_mb),
            ]

            for name, mb in breakdown_items:
                pct = (mb / total_known * 100.0) if total_known > 0 else 0.0
                table.add_row(name, f"{mb:,.1f}", f"{pct:.1f}%")

            table.add_row(
                Text("Total", style="bold"),
                Text(f"{total_known:,.1f}", style="bold"),
                Text("100.0%", style="bold"),
            )

            console.print(
                Panel(
                    table,
                    title="[bold]Memory Breakdown[/bold]",
                    border_style="blue",
                    padding=(0, 1),
                )
            )
        else:
            # No breakdown available; show snapshot-level info.
            snapshots = self._memory_result.snapshots
            if snapshots:
                peak = max(snapshots, key=lambda s: s.allocated_mb)
                info_text = Text()
                info_text.append(f"Peak Allocated: {peak.allocated_mb:,.1f} MB\n")
                info_text.append(f"Peak Reserved:  {peak.reserved_mb:,.1f} MB\n")
                info_text.append(f"Snapshots:      {len(snapshots)}")

                console.print(
                    Panel(
                        info_text,
                        title="[bold]Memory Summary[/bold]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                )

        # Leak and OOM risk info
        risk_lines: List[str] = []
        if self._memory_result.leak_detected:
            risk_lines.append(
                f"[red bold]Memory leak detected![/red bold] "
                f"({len(self._memory_result.leak_points)} leak point(s))"
            )

        oom = self._memory_result.oom_risk
        if oom.peak_usage_pct > 0:
            risk_color = _efficiency_color(100.0 - oom.peak_usage_pct)
            risk_lines.append(
                f"OOM Risk Score: [{risk_color}]{oom.risk_score:.2f}[/{risk_color}]  "
                f"Peak Usage: [{risk_color}]{oom.peak_usage_pct:.1f}%[/{risk_color}]  "
                f"Headroom: {oom.headroom_mb:,.0f} MB"
            )

        if risk_lines:
            for line in risk_lines:
                console.print(f"  {line}")

    # ==================================================================
    # HTML report
    # ==================================================================

    def generate_html_report(self, output_path: str) -> None:
        """Generate a self-contained HTML report with inline CSS and JS.

        The report includes interactive canvas-based charts for time breakdown,
        GPU utilization, and memory usage, plus tables for kernel analysis and
        bottleneck recommendations.

        Parameters
        ----------
        output_path:
            File path to write the HTML report to.
        """
        summary = self._summary
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        efficiency_pct = summary.overall_efficiency * 100.0

        # Build chart data
        time_breakdown_data = self._build_time_breakdown_data()
        gpu_timeline_data = self._build_gpu_timeline_data()
        memory_timeline_data = self._build_memory_timeline_data()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chamber Profiler Report</title>
<style>
{self._get_css()}
</style>
</head>
<body>

<div class="container">

<!-- Header -->
<header class="header">
    <h1>Chamber Profiler</h1>
    <p class="subtitle">Performance Analysis Report</p>
    <div class="meta">
        <span>Generated: {html.escape(now)}</span>
        <span>Duration: {html.escape(_format_duration(summary.total_time_s))}</span>
        <span>Primary Bottleneck: {html.escape(summary.primary_bottleneck.replace('_', ' ').title())}</span>
        {self._gpu_count_html()}
    </div>
</header>

<!-- Efficiency Score -->
<section class="card">
    <h2>Overall GPU Efficiency</h2>
    <div class="efficiency-container">
        <div class="efficiency-bar-bg">
            <div class="efficiency-bar-fill" style="width: {efficiency_pct:.1f}%; background: {_efficiency_hex(efficiency_pct)};"></div>
        </div>
        <div class="efficiency-value" style="color: {_efficiency_hex(efficiency_pct)};">{efficiency_pct:.1f}%</div>
    </div>
</section>

<!-- Time Breakdown -->
<section class="card">
    <h2>Time Breakdown</h2>
    <div class="chart-row">
        <div class="chart-container">
            <canvas id="timeBreakdownChart" width="400" height="300"></canvas>
        </div>
        <div class="breakdown-table-container">
            {self._build_time_breakdown_table_html()}
        </div>
    </div>
</section>

<!-- GPU Utilization Over Time -->
{self._build_gpu_chart_section_html()}

<!-- Memory Usage Over Time -->
{self._build_memory_chart_section_html()}

<!-- Kernel Breakdown -->
{self._build_kernel_section_html()}

<!-- Bottleneck Analysis -->
{self._build_bottleneck_section_html()}

<!-- Recommendations -->
{self._build_recommendations_section_html()}

<!-- Summary -->
<section class="card">
    <h2>Summary</h2>
    <p class="summary-text">{html.escape(summary.summary_text)}</p>
</section>

</div>

<script>
{self._get_js(time_breakdown_data, gpu_timeline_data, memory_timeline_data)}
</script>

</body>
</html>"""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)

    def _gpu_count_html(self) -> str:
        """Return an HTML span for GPU count if available."""
        if self._gpu_result is not None:
            return f"<span>GPUs: {self._gpu_result.gpu_count}</span>"
        return ""

    def _build_time_breakdown_data(self) -> List[Dict[str, Any]]:
        """Return time breakdown data for the chart."""
        s = self._summary
        return [
            {"label": "Compute", "value": round(s.compute_pct, 1), "color": "#22d3ee"},
            {"label": "Memory", "value": round(s.memory_pct, 1), "color": "#60a5fa"},
            {"label": "Communication", "value": round(s.communication_pct, 1), "color": "#c084fc"},
            {"label": "Data Loading", "value": round(s.data_loading_pct, 1), "color": "#facc15"},
            {"label": "Idle", "value": round(s.idle_pct, 1), "color": "#f87171"},
        ]

    def _build_gpu_timeline_data(self) -> List[Dict[str, float]]:
        """Return GPU utilization timeline data for the chart."""
        if self._gpu_result is None or not self._gpu_result.snapshots:
            return []

        snapshots = self._gpu_result.snapshots
        if not snapshots:
            return []

        base_ts = snapshots[0].timestamp
        data: List[Dict[str, float]] = []

        # Downsample if too many points.
        step = max(1, len(snapshots) // 200)
        for i in range(0, len(snapshots), step):
            s = snapshots[i]
            data.append({
                "t": round(s.timestamp - base_ts, 2),
                "util": round(s.utilization_pct, 1),
                "sm": round(s.sm_activity_pct, 1),
                "mem_bw": round(s.memory_bandwidth_pct, 1),
            })

        return data

    def _build_memory_timeline_data(self) -> List[Dict[str, float]]:
        """Return memory usage timeline data for the chart."""
        if self._memory_result is None or not self._memory_result.snapshots:
            return []

        snapshots = self._memory_result.snapshots
        if not snapshots:
            return []

        base_ts = snapshots[0].timestamp
        data: List[Dict[str, float]] = []

        step = max(1, len(snapshots) // 200)
        for i in range(0, len(snapshots), step):
            s = snapshots[i]
            data.append({
                "t": round(s.timestamp - base_ts, 2),
                "allocated": round(s.allocated_mb, 1),
                "reserved": round(s.reserved_mb, 1),
            })

        return data

    def _build_time_breakdown_table_html(self) -> str:
        """Build the time breakdown table HTML."""
        s = self._summary
        rows = [
            ("Compute", s.compute_pct, "#22d3ee"),
            ("Memory", s.memory_pct, "#60a5fa"),
            ("Communication", s.communication_pct, "#c084fc"),
            ("Data Loading", s.data_loading_pct, "#facc15"),
            ("Idle", s.idle_pct, "#f87171"),
        ]

        html_parts = ['<table class="data-table">']
        html_parts.append(
            "<thead><tr>"
            "<th>Category</th><th>%</th><th>Time</th>"
            "</tr></thead><tbody>"
        )

        for name, pct, color in rows:
            cat_time = s.total_time_s * (pct / 100.0)
            html_parts.append(
                f'<tr>'
                f'<td><span class="color-dot" style="background:{color};"></span>{html.escape(name)}</td>'
                f'<td>{pct:.1f}%</td>'
                f'<td>{html.escape(_format_duration(cat_time))}</td>'
                f'</tr>'
            )

        html_parts.append("</tbody></table>")
        return "\n".join(html_parts)

    def _build_gpu_chart_section_html(self) -> str:
        """Build the GPU utilization chart section HTML."""
        if self._gpu_result is None or not self._gpu_result.snapshots:
            return ""

        snapshots = self._gpu_result.snapshots
        avg_util = sum(s.utilization_pct for s in snapshots) / len(snapshots)
        peak_mem = max(s.memory_used_mb for s in snapshots)
        total_mem = max(s.memory_total_mb for s in snapshots) if snapshots else 0.0
        avg_power = sum(s.power_watts for s in snapshots) / len(snapshots)
        avg_temp = sum(s.temperature_c for s in snapshots) / len(snapshots)

        return f"""
<section class="card">
    <h2>GPU Utilization Over Time</h2>
    <div class="gpu-stats">
        <div class="stat-box">
            <div class="stat-value" style="color: {_efficiency_hex(avg_util)};">{avg_util:.1f}%</div>
            <div class="stat-label">Avg Utilization</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{peak_mem:,.0f} MB</div>
            <div class="stat-label">Peak Memory</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_power:.1f} W</div>
            <div class="stat-label">Avg Power</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_temp:.1f} C</div>
            <div class="stat-label">Avg Temperature</div>
        </div>
    </div>
    <div class="chart-container-full">
        <canvas id="gpuUtilChart" width="800" height="250"></canvas>
    </div>
</section>"""

    def _build_memory_chart_section_html(self) -> str:
        """Build the memory usage chart section HTML."""
        if self._memory_result is None or not self._memory_result.snapshots:
            return ""

        snapshots = self._memory_result.snapshots
        peak = max(snapshots, key=lambda s: s.allocated_mb)

        bd_html = ""
        bd = self._memory_result.breakdown
        if bd is not None:
            total_known = (
                bd.parameter_memory_mb + bd.gradient_memory_mb
                + bd.optimizer_state_mb + bd.activation_memory_mb + bd.other_mb
            )
            items = [
                ("Parameters", bd.parameter_memory_mb),
                ("Gradients", bd.gradient_memory_mb),
                ("Optimizer State", bd.optimizer_state_mb),
                ("Activations", bd.activation_memory_mb),
                ("Other", bd.other_mb),
            ]
            bd_html = '<table class="data-table"><thead><tr><th>Category</th><th>MB</th><th>%</th></tr></thead><tbody>'
            for name, mb in items:
                pct = (mb / total_known * 100.0) if total_known > 0 else 0.0
                bd_html += f"<tr><td>{html.escape(name)}</td><td>{mb:,.1f}</td><td>{pct:.1f}%</td></tr>"
            bd_html += f'<tr class="total-row"><td><strong>Total</strong></td><td><strong>{total_known:,.1f}</strong></td><td><strong>100.0%</strong></td></tr>'
            bd_html += "</tbody></table>"

        leak_html = ""
        if self._memory_result.leak_detected:
            leak_html = f'<div class="alert alert-danger">Memory leak detected! ({len(self._memory_result.leak_points)} leak point(s))</div>'

        oom = self._memory_result.oom_risk
        oom_html = ""
        if oom.peak_usage_pct > 0:
            oom_html = (
                f'<div class="oom-info">'
                f'<span>OOM Risk: <strong style="color:{_efficiency_hex(100.0 - oom.peak_usage_pct)};">'
                f'{oom.risk_score:.2f}</strong></span>'
                f'<span>Peak Usage: <strong>{oom.peak_usage_pct:.1f}%</strong></span>'
                f'<span>Headroom: <strong>{oom.headroom_mb:,.0f} MB</strong></span>'
                f'</div>'
            )

        return f"""
<section class="card">
    <h2>Memory Usage</h2>
    {leak_html}
    {oom_html}
    <div class="chart-row">
        <div class="chart-container-full">
            <canvas id="memoryChart" width="800" height="250"></canvas>
        </div>
    </div>
    <div class="memory-details">
        <div class="stat-box">
            <div class="stat-value">{peak.allocated_mb:,.1f} MB</div>
            <div class="stat-label">Peak Allocated</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{peak.reserved_mb:,.1f} MB</div>
            <div class="stat-label">Peak Reserved</div>
        </div>
    </div>
    {bd_html}
</section>"""

    def _build_kernel_section_html(self) -> str:
        """Build the kernel breakdown table HTML section."""
        if self._kernel_result is None or not self._kernel_result.top_kernels:
            return ""

        top_kernels = self._kernel_result.top_kernels[:10]
        total_us = self._kernel_result.total_kernel_time_us

        rows_html = ""
        for i, k in enumerate(top_kernels, start=1):
            pct = (k.duration_us / total_us * 100.0) if total_us > 0 else 0.0
            occ_str = "-"
            if k.occupancy is not None:
                occ_pct = k.occupancy * 100.0
                occ_str = f'<span style="color:{_efficiency_hex(occ_pct)};">{occ_pct:.1f}%</span>'

            rows_html += (
                f"<tr>"
                f"<td>{i}</td>"
                f"<td class='kernel-name'>{html.escape(_truncate_kernel_name(k.name, 50))}</td>"
                f"<td>{html.escape(k.category)}</td>"
                f"<td>{k.duration_us:,.1f}</td>"
                f"<td>{pct:.1f}%</td>"
                f"<td>{occ_str}</td>"
                f"</tr>"
            )

        # Category breakdown summary
        cat_html = ""
        if self._kernel_result.category_breakdown:
            cat_html = '<div class="category-breakdown"><h3>Kernel Category Breakdown</h3><div class="cat-bars">'
            for cat, us in sorted(
                self._kernel_result.category_breakdown.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = (us / total_us * 100.0) if total_us > 0 else 0.0
                if pct < 0.1:
                    continue
                cat_html += (
                    f'<div class="cat-bar-row">'
                    f'<span class="cat-bar-label">{html.escape(cat)}</span>'
                    f'<div class="cat-bar-bg"><div class="cat-bar-fill" style="width:{pct:.1f}%;"></div></div>'
                    f'<span class="cat-bar-value">{pct:.1f}%</span>'
                    f'</div>'
                )
            cat_html += "</div></div>"

        return f"""
<section class="card">
    <h2>Kernel Analysis</h2>
    {cat_html}
    <table class="data-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Kernel Name</th>
                <th>Category</th>
                <th>Duration (us)</th>
                <th>% of Total</th>
                <th>Occupancy</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</section>"""

    def _build_bottleneck_section_html(self) -> str:
        """Build the bottleneck analysis section HTML."""
        if not self._summary.bottlenecks:
            return ""

        rows_html = ""
        for i, b in enumerate(self._summary.bottlenecks[:5], start=1):
            impact_pct = b.impact_score * 100.0
            category_label = b.category.value.replace("_", " ").title()

            recs_html = "<ul>"
            for rec in b.recommendations[:3]:
                recs_html += f"<li>{html.escape(rec)}</li>"
            recs_html += "</ul>"

            rows_html += (
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>{html.escape(category_label)}</td>"
                f'<td><span style="color:{_efficiency_hex(100.0 - impact_pct)};">{impact_pct:.0f}%</span></td>'
                f"<td>{html.escape(b.description)}</td>"
                f"<td>{b.expected_speedup or '-'}</td>"
                f"</tr>"
                f"<tr class='rec-row'>"
                f"<td></td>"
                f'<td colspan="4">{recs_html}</td>'
                f"</tr>"
            )

        return f"""
<section class="card">
    <h2>Bottleneck Analysis</h2>
    <table class="data-table bottleneck-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Category</th>
                <th>Impact</th>
                <th>Description</th>
                <th>Expected Speedup</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</section>"""

    def _build_recommendations_section_html(self) -> str:
        """Build the recommendations section HTML."""
        if not self._summary.bottlenecks:
            return ""

        # Collect all unique recommendations across all bottlenecks.
        seen: set[str] = set()
        recs: List[Dict[str, str]] = []
        for b in self._summary.bottlenecks:
            for rec in b.recommendations:
                if rec not in seen:
                    seen.add(rec)
                    recs.append({
                        "category": b.category.value.replace("_", " ").title(),
                        "text": rec,
                        "speedup": b.expected_speedup,
                    })

        if not recs:
            return ""

        items_html = ""
        for i, rec in enumerate(recs[:10], start=1):
            items_html += (
                f'<div class="rec-item">'
                f'<div class="rec-number">{i}</div>'
                f'<div class="rec-content">'
                f'<div class="rec-category">{html.escape(rec["category"])}</div>'
                f'<div class="rec-text">{html.escape(rec["text"])}</div>'
                f'</div>'
                f'</div>'
            )

        return f"""
<section class="card">
    <h2>Recommendations</h2>
    <div class="recommendations-list">
        {items_html}
    </div>
</section>"""

    @staticmethod
    def _get_css() -> str:
        """Return the inline CSS for the HTML report."""
        return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.header {
    text-align: center;
    padding: 40px 20px 30px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}
.header h1 {
    font-size: 2.2em;
    background: linear-gradient(135deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.subtitle { color: #94a3b8; font-size: 1.1em; margin-bottom: 16px; }
.meta { display: flex; gap: 24px; justify-content: center; flex-wrap: wrap; color: #94a3b8; font-size: 0.9em; }
.card {
    background: #1e293b;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.card h2 {
    font-size: 1.3em;
    margin-bottom: 16px;
    color: #f1f5f9;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
}
.card h3 { font-size: 1.05em; margin-bottom: 10px; color: #cbd5e1; }
.efficiency-container { text-align: center; padding: 10px 0; }
.efficiency-bar-bg {
    width: 100%;
    height: 28px;
    background: #334155;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 8px;
}
.efficiency-bar-fill {
    height: 100%;
    border-radius: 14px;
    transition: width 0.6s ease;
}
.efficiency-value { font-size: 2em; font-weight: 700; }
.chart-row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.chart-container { flex: 1; min-width: 300px; }
.chart-container-full { width: 100%; }
.breakdown-table-container { flex: 1; min-width: 250px; }
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}
.data-table th {
    text-align: left;
    padding: 10px 12px;
    background: #0f172a;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8em;
    letter-spacing: 0.5px;
}
.data-table td { padding: 10px 12px; border-bottom: 1px solid #334155; }
.data-table tbody tr:hover { background: #273549; }
.data-table .total-row td { border-top: 2px solid #475569; background: #0f172a; }
.kernel-name { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85em; word-break: break-all; }
.color-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.gpu-stats, .memory-details { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
.stat-box {
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 12px;
    background: #0f172a;
    border-radius: 8px;
    border: 1px solid #334155;
}
.stat-value { font-size: 1.5em; font-weight: 700; color: #f1f5f9; }
.stat-label { font-size: 0.8em; color: #94a3b8; margin-top: 4px; }
.category-breakdown { margin-bottom: 20px; }
.cat-bars { display: flex; flex-direction: column; gap: 6px; }
.cat-bar-row { display: flex; align-items: center; gap: 10px; }
.cat-bar-label { width: 120px; text-align: right; font-size: 0.85em; color: #94a3b8; }
.cat-bar-bg { flex: 1; height: 18px; background: #334155; border-radius: 9px; overflow: hidden; }
.cat-bar-fill { height: 100%; background: linear-gradient(90deg, #60a5fa, #c084fc); border-radius: 9px; }
.cat-bar-value { width: 50px; font-size: 0.85em; color: #cbd5e1; }
.bottleneck-table .rec-row td { padding-top: 0; border-bottom: 2px solid #334155; }
.bottleneck-table .rec-row ul { margin: 4px 0 8px 20px; font-size: 0.85em; color: #94a3b8; }
.bottleneck-table .rec-row li { margin-bottom: 4px; }
.alert { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; }
.alert-danger { background: rgba(248, 113, 113, 0.15); border: 1px solid #f87171; color: #fca5a5; }
.oom-info { display: flex; gap: 24px; margin-bottom: 16px; font-size: 0.9em; flex-wrap: wrap; }
.recommendations-list { display: flex; flex-direction: column; gap: 12px; }
.rec-item { display: flex; gap: 14px; padding: 14px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; }
.rec-number {
    width: 28px; height: 28px;
    background: #334155;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85em; color: #60a5fa;
    flex-shrink: 0;
}
.rec-content { flex: 1; }
.rec-category { font-size: 0.75em; color: #c084fc; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.rec-text { font-size: 0.9em; color: #cbd5e1; }
.summary-text { font-size: 1.05em; color: #cbd5e1; line-height: 1.8; }
canvas { display: block; width: 100% !important; }
"""

    @staticmethod
    def _get_js(
        time_breakdown_data: List[Dict[str, Any]],
        gpu_timeline_data: List[Dict[str, float]],
        memory_timeline_data: List[Dict[str, float]],
    ) -> str:
        """Return the inline JavaScript for the HTML report charts."""
        tb_json = json.dumps(time_breakdown_data)
        gpu_json = json.dumps(gpu_timeline_data)
        mem_json = json.dumps(memory_timeline_data)

        return f"""
(function() {{
    'use strict';

    var timeBreakdownData = {tb_json};
    var gpuTimelineData = {gpu_json};
    var memoryTimelineData = {mem_json};

    // ---- Utility ----
    function getCtx(id) {{
        var el = document.getElementById(id);
        if (!el) return null;
        // Set actual pixel size from CSS layout
        var rect = el.getBoundingClientRect();
        el.width = rect.width * (window.devicePixelRatio || 1);
        el.height = rect.height * (window.devicePixelRatio || 1);
        var ctx = el.getContext('2d');
        ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
        return {{ ctx: ctx, w: rect.width, h: rect.height }};
    }}

    // ---- Time Breakdown Bar Chart ----
    function drawTimeBreakdown() {{
        var c = getCtx('timeBreakdownChart');
        if (!c) return;
        var ctx = c.ctx, w = c.w, h = c.h;
        var data = timeBreakdownData.filter(function(d) {{ return d.value > 0; }});
        if (data.length === 0) return;

        var padding = {{ top: 20, right: 20, bottom: 40, left: 60 }};
        var chartW = w - padding.left - padding.right;
        var chartH = h - padding.top - padding.bottom;
        var maxVal = Math.max.apply(null, data.map(function(d) {{ return d.value; }}));
        maxVal = Math.max(maxVal, 1);

        var barWidth = Math.min(chartW / data.length * 0.6, 60);
        var gap = (chartW - barWidth * data.length) / (data.length + 1);

        // Grid lines
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 0.5;
        ctx.font = '11px -apple-system, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'right';
        for (var i = 0; i <= 4; i++) {{
            var y = padding.top + chartH - (i / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();
            ctx.fillText((maxVal * i / 4).toFixed(0) + '%', padding.left - 8, y + 4);
        }}

        // Bars
        ctx.textAlign = 'center';
        for (var j = 0; j < data.length; j++) {{
            var d = data[j];
            var x = padding.left + gap + j * (barWidth + gap);
            var barH = (d.value / maxVal) * chartH;
            var y2 = padding.top + chartH - barH;

            // Bar with rounded top
            ctx.fillStyle = d.color;
            ctx.beginPath();
            var r = Math.min(4, barWidth / 2);
            ctx.moveTo(x, padding.top + chartH);
            ctx.lineTo(x, y2 + r);
            ctx.quadraticCurveTo(x, y2, x + r, y2);
            ctx.lineTo(x + barWidth - r, y2);
            ctx.quadraticCurveTo(x + barWidth, y2, x + barWidth, y2 + r);
            ctx.lineTo(x + barWidth, padding.top + chartH);
            ctx.closePath();
            ctx.fill();

            // Value on top
            ctx.fillStyle = '#e2e8f0';
            ctx.font = 'bold 11px -apple-system, sans-serif';
            ctx.fillText(d.value.toFixed(1) + '%', x + barWidth / 2, y2 - 6);

            // Label below
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.fillText(d.label, x + barWidth / 2, padding.top + chartH + 18);
        }}
    }}

    // ---- Line Chart Utility ----
    function drawLineChart(canvasId, data, series, yLabel) {{
        var c = getCtx(canvasId);
        if (!c || data.length === 0) return;
        var ctx = c.ctx, w = c.w, h = c.h;

        var padding = {{ top: 20, right: 20, bottom: 40, left: 55 }};
        var chartW = w - padding.left - padding.right;
        var chartH = h - padding.top - padding.bottom;

        var maxT = Math.max.apply(null, data.map(function(d) {{ return d.t; }}));
        var maxVal = 0;
        for (var s = 0; s < series.length; s++) {{
            for (var i2 = 0; i2 < data.length; i2++) {{
                var v = data[i2][series[s].key];
                if (v > maxVal) maxVal = v;
            }}
        }}
        maxVal = Math.max(maxVal * 1.1, 1);
        maxT = Math.max(maxT, 0.01);

        // Grid
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 0.5;
        ctx.font = '10px -apple-system, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'right';
        for (var g = 0; g <= 4; g++) {{
            var gy = padding.top + chartH - (g / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, gy);
            ctx.lineTo(w - padding.right, gy);
            ctx.stroke();
            ctx.fillText((maxVal * g / 4).toFixed(0), padding.left - 6, gy + 4);
        }}

        // Y-axis label
        ctx.save();
        ctx.translate(12, padding.top + chartH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();

        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', padding.left + chartW / 2, h - 6);

        // X ticks
        for (var xt = 0; xt <= 4; xt++) {{
            var xpos = padding.left + (xt / 4) * chartW;
            ctx.fillText((maxT * xt / 4).toFixed(1), xpos, padding.top + chartH + 18);
        }}

        // Lines
        for (var si = 0; si < series.length; si++) {{
            var ser = series[si];
            ctx.beginPath();
            ctx.strokeStyle = ser.color;
            ctx.lineWidth = 1.5;
            for (var di = 0; di < data.length; di++) {{
                var px = padding.left + (data[di].t / maxT) * chartW;
                var py = padding.top + chartH - (data[di][ser.key] / maxVal) * chartH;
                if (di === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Fill area
            ctx.globalAlpha = 0.08;
            ctx.lineTo(padding.left + (data[data.length-1].t / maxT) * chartW, padding.top + chartH);
            ctx.lineTo(padding.left + (data[0].t / maxT) * chartW, padding.top + chartH);
            ctx.closePath();
            ctx.fillStyle = ser.color;
            ctx.fill();
            ctx.globalAlpha = 1.0;
        }}

        // Legend
        var legendX = padding.left + 10;
        var legendY = padding.top + 6;
        ctx.font = '11px -apple-system, sans-serif';
        for (var li = 0; li < series.length; li++) {{
            ctx.fillStyle = series[li].color;
            ctx.fillRect(legendX, legendY + li * 18, 12, 12);
            ctx.fillStyle = '#cbd5e1';
            ctx.textAlign = 'left';
            ctx.fillText(series[li].label, legendX + 18, legendY + li * 18 + 10);
        }}
    }}

    // ---- GPU Utilization Chart ----
    function drawGpuChart() {{
        drawLineChart('gpuUtilChart', gpuTimelineData, [
            {{ key: 'util', color: '#22d3ee', label: 'GPU Utilization (%)' }},
            {{ key: 'sm', color: '#60a5fa', label: 'SM Activity (%)' }},
            {{ key: 'mem_bw', color: '#c084fc', label: 'Mem Bandwidth (%)' }}
        ], 'Utilization (%)');
    }}

    // ---- Memory Chart ----
    function drawMemoryChart() {{
        drawLineChart('memoryChart', memoryTimelineData, [
            {{ key: 'allocated', color: '#60a5fa', label: 'Allocated (MB)' }},
            {{ key: 'reserved', color: '#94a3b8', label: 'Reserved (MB)' }}
        ], 'Memory (MB)');
    }}

    // ---- Initialize ----
    function init() {{
        drawTimeBreakdown();
        drawGpuChart();
        drawMemoryChart();
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', init);
    }} else {{
        init();
    }}

    window.addEventListener('resize', function() {{
        setTimeout(init, 100);
    }});
}})();
"""

    # ==================================================================
    # JSON report
    # ==================================================================

    def generate_json_report(self, output_path: str) -> None:
        """Export all profiling data and analysis as structured JSON.

        The JSON output includes the performance summary, all raw profiling
        results (GPU, kernel, memory, communication, data loading), and
        metadata about the report.

        Parameters
        ----------
        output_path:
            File path to write the JSON report to.
        """
        data: Dict[str, Any] = {
            "metadata": {
                "tool": "chamber-profiler",
                "report_generated": datetime.datetime.now().isoformat(),
                "format_version": "1.0",
            },
            "summary": self._summary.to_dict(),
        }

        # Raw profiler results
        if self._gpu_result is not None:
            data["gpu_profile"] = self._gpu_result.to_dict()

        if self._kernel_result is not None:
            data["kernel_trace"] = self._kernel_result.to_dict()

        if self._memory_result is not None:
            data["memory_analysis"] = self._memory_result.to_dict()

        if self._comm_result is not None:
            data["communication_profile"] = self._comm_result.to_dict()

        if self._data_result is not None:
            data["data_loading_profile"] = self._data_result.to_dict()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON report written to %s", output_path)
