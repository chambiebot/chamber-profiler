"""Flame graph data generator from GPU kernel traces.

Converts kernel trace data into the collapsed stack format used by
Brendan Gregg's FlameGraph tools and compatible viewers (speedscope,
inferno, etc.).

Each line represents a stack frame path with a sample count (duration in
microseconds), e.g.::

    model;encoder;attention;flash_fwd_kernel 4500
    model;encoder;ffn;cublasSgemm 3200

The module can also generate a simple SVG flame graph without external
dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.profiler.kernel_tracer import KernelTraceResult, KernelRecord

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class FlameGraphFrame:
    """A single frame in the flame graph."""

    name: str
    duration_us: float
    children: Dict[str, "FlameGraphFrame"] = field(default_factory=dict)
    self_time_us: float = 0.0

    def total_time_us(self) -> float:
        return self.self_time_us + sum(
            c.total_time_us() for c in self.children.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_us": self.duration_us,
            "self_time_us": self.self_time_us,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


@dataclass
class FlameGraphData:
    """Output of the flame graph generator."""

    collapsed_stacks: str
    root: FlameGraphFrame
    total_time_us: float
    num_stacks: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collapsed_stacks": self.collapsed_stacks,
            "root": self.root.to_dict(),
            "total_time_us": self.total_time_us,
            "num_stacks": self.num_stacks,
        }


# ============================================================================
# Stack path builders
# ============================================================================

# Map kernel categories to logical groupings for the flame graph.
_CATEGORY_GROUP: Dict[str, str] = {
    "gemm": "compute/matmul",
    "conv": "compute/conv",
    "attention": "compute/attention",
    "elementwise": "compute/elementwise",
    "reduction": "compute/reduction",
    "memory": "memory",
    "communication": "communication",
    "other": "compute/other",
}


def _build_stack_path(kernel: KernelRecord) -> str:
    """Build a semicolon-separated stack path for a kernel.

    The path hierarchy is::

        category_group;layer_name;kernel_name

    If the kernel has no layer name, the middle level is omitted.
    """
    group = _CATEGORY_GROUP.get(kernel.category, "other")

    parts: List[str] = [group]

    if kernel.layer_name:
        # Split dotted layer names into nested frames.
        layer_parts = kernel.layer_name.split(".")
        parts.extend(layer_parts)

    # Simplify the kernel name: strip "void " prefix and template args.
    name = kernel.name
    if name.startswith("void "):
        name = name[5:]
    # Truncate at first '<' to remove template parameters.
    lt_idx = name.find("<")
    if lt_idx > 0:
        name = name[:lt_idx]
    # Truncate at first '(' to remove function arguments.
    paren_idx = name.find("(")
    if paren_idx > 0:
        name = name[:paren_idx]

    parts.append(name.strip())

    return ";".join(parts)


# ============================================================================
# Flame Graph Generator
# ============================================================================


class FlameGraphGenerator:
    """Generate flame graph data from GPU kernel traces.

    Usage::

        generator = FlameGraphGenerator(kernel_result=kernel_result)
        data = generator.generate()
        # Write collapsed stacks to a file
        generator.write_collapsed(data, "profile.folded")
        # Or generate an SVG
        generator.write_svg(data, "profile.svg")
    """

    def __init__(
        self,
        kernel_result: Optional[KernelTraceResult] = None,
        min_duration_us: float = 0.0,
    ) -> None:
        self._kernel_result = kernel_result
        self._min_duration_us = min_duration_us

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> FlameGraphData:
        """Generate flame graph data from kernel traces.

        Returns
        -------
        FlameGraphData
            The flame graph data including collapsed stacks and a tree.
        """
        if self._kernel_result is None or not self._kernel_result.kernels:
            return FlameGraphData(
                collapsed_stacks="",
                root=FlameGraphFrame(name="root", duration_us=0.0),
                total_time_us=0.0,
                num_stacks=0,
            )

        kernels = self._kernel_result.kernels
        if self._min_duration_us > 0:
            kernels = [
                k for k in kernels if k.duration_us >= self._min_duration_us
            ]

        # Build collapsed stacks.
        stack_counts: Dict[str, float] = {}
        for kernel in kernels:
            path = _build_stack_path(kernel)
            stack_counts[path] = stack_counts.get(path, 0.0) + kernel.duration_us

        # Build collapsed stacks string.
        lines: List[str] = []
        for path, count in sorted(
            stack_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"{path} {int(count)}")

        collapsed = "\n".join(lines)

        # Build tree.
        root = FlameGraphFrame(name="root", duration_us=0.0)
        for path, count in stack_counts.items():
            self._insert_into_tree(root, path, count)

        total_time = sum(stack_counts.values())
        root.duration_us = total_time

        return FlameGraphData(
            collapsed_stacks=collapsed,
            root=root,
            total_time_us=total_time,
            num_stacks=len(stack_counts),
        )

    def write_collapsed(self, data: FlameGraphData, output_path: str) -> str:
        """Write collapsed stacks to a file.

        Parameters
        ----------
        data:
            The flame graph data from :meth:`generate`.
        output_path:
            File path for the output.

        Returns
        -------
        str
            The output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.collapsed_stacks, encoding="utf-8")
        logger.info(
            "Collapsed stacks written to %s (%d stacks)",
            output_path,
            data.num_stacks,
        )
        return output_path

    def write_svg(
        self,
        data: FlameGraphData,
        output_path: str,
        title: str = "GPU Kernel Flame Graph",
        width: int = 1200,
    ) -> str:
        """Generate a simple SVG flame graph.

        Parameters
        ----------
        data:
            The flame graph data from :meth:`generate`.
        output_path:
            File path for the SVG output.
        title:
            Title displayed at the top of the SVG.
        width:
            SVG width in pixels.

        Returns
        -------
        str
            The output file path.
        """
        svg = self._render_svg(data, title=title, width=width)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
        logger.info("SVG flame graph written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_into_tree(
        root: FlameGraphFrame, path: str, duration_us: float
    ) -> None:
        """Insert a stack path into the tree."""
        parts = path.split(";")
        node = root
        for part in parts:
            if part not in node.children:
                node.children[part] = FlameGraphFrame(
                    name=part, duration_us=0.0
                )
            node = node.children[part]
            node.duration_us += duration_us
        # The leaf gets the self-time.
        node.self_time_us += duration_us

    @staticmethod
    def _render_svg(
        data: FlameGraphData,
        title: str = "GPU Kernel Flame Graph",
        width: int = 1200,
    ) -> str:
        """Render a minimal SVG flame graph from collapsed stacks."""
        row_height = 18
        font_size = 11
        padding = 10
        title_height = 30

        # Parse collapsed stacks into (path_parts, count) tuples.
        entries: List[Tuple[List[str], int]] = []
        for line in data.collapsed_stacks.strip().splitlines():
            if not line.strip():
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            stack_path = parts[0].split(";")
            try:
                count = int(parts[1])
            except ValueError:
                continue
            entries.append((stack_path, count))

        if not entries:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="60">'
                f'<text x="10" y="30" font-size="14">No data</text></svg>'
            )

        total = sum(c for _, c in entries)
        if total <= 0:
            total = 1

        # Compute max depth.
        max_depth = max(len(p) for p, _ in entries)
        height = title_height + (max_depth + 1) * row_height + padding * 2

        # Color map for categories.
        colors = {
            "compute": "#e8912d",
            "memory": "#4a90d9",
            "communication": "#50b848",
            "other": "#999999",
        }

        def _get_color(name: str) -> str:
            for key, color in colors.items():
                if key in name.lower():
                    return color
            return "#dd7755"

        rects: List[str] = []

        # Build rectangles from collapsed stacks — simple left-to-right layout.
        x_cursor = 0.0
        usable_width = width - 2 * padding

        for path_parts, count in entries:
            w = (count / total) * usable_width
            if w < 0.5:
                x_cursor += w
                continue

            for depth, frame_name in enumerate(path_parts):
                y = height - padding - (depth + 1) * row_height
                color = _get_color(frame_name)

                # Truncate label if box is too small.
                label = frame_name
                max_chars = int(w / (font_size * 0.6))
                if max_chars < 3:
                    label = ""
                elif len(label) > max_chars:
                    label = label[: max_chars - 2] + ".."

                rects.append(
                    f'<g>'
                    f'<rect x="{padding + x_cursor:.1f}" y="{y:.1f}" '
                    f'width="{w:.1f}" height="{row_height - 1}" '
                    f'fill="{color}" rx="1" />'
                )
                if label:
                    rects.append(
                        f'<text x="{padding + x_cursor + 2:.1f}" '
                        f'y="{y + row_height - 5:.1f}" '
                        f'font-size="{font_size}" font-family="monospace" '
                        f'fill="white">{_svg_escape(label)}</text>'
                    )
                rects.append("</g>")

            x_cursor += w

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{int(height)}">',
            f'<style>rect:hover {{ opacity: 0.8; }}</style>',
            f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
            f'<text x="{width // 2}" y="20" text-anchor="middle" '
            f'font-size="14" font-family="sans-serif" fill="white">'
            f'{_svg_escape(title)}</text>',
            *rects,
            "</svg>",
        ]

        return "\n".join(svg_parts)


def _svg_escape(text: str) -> str:
    """Escape special characters for SVG text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
