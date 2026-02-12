"""Tests for src.profiler.flame_graph."""

import os
import tempfile

import pytest

from src.profiler.flame_graph import (
    FlameGraphFrame,
    FlameGraphData,
    FlameGraphGenerator,
    _build_stack_path,
    _CATEGORY_GROUP,
    _svg_escape,
)
from src.profiler.kernel_tracer import KernelRecord, KernelTraceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_kernel(
    name: str = "volta_sgemm_128x64",
    duration_us: float = 100.0,
    category: str = "gemm",
    layer_name: str = None,
) -> KernelRecord:
    return KernelRecord(
        name=name,
        duration_us=duration_us,
        grid_size=(128, 1, 1),
        block_size=(256, 1, 1),
        shared_memory_bytes=0,
        device_index=0,
        category=category,
        layer_name=layer_name,
    )


def _make_kernel_result(kernels=None) -> KernelTraceResult:
    if kernels is None:
        kernels = [
            _make_kernel("sgemm_kernel", 500.0, "gemm", "model.linear"),
            _make_kernel("flash_fwd_kernel", 300.0, "attention", "model.attn"),
            _make_kernel("elementwise_add", 100.0, "elementwise"),
            _make_kernel("nccl_allreduce", 200.0, "communication"),
            _make_kernel("memcpy_h2d", 50.0, "memory"),
        ]
    return KernelTraceResult(
        kernels=kernels,
        total_kernel_time_us=sum(k.duration_us for k in kernels),
        top_kernels=kernels[:3],
        inefficient_kernels=[],
        category_breakdown={},
    )


# ---------------------------------------------------------------------------
# FlameGraphFrame
# ---------------------------------------------------------------------------


class TestFlameGraphFrame:
    def test_creation(self):
        frame = FlameGraphFrame(name="root", duration_us=100.0)
        assert frame.name == "root"
        assert frame.duration_us == 100.0
        assert frame.children == {}
        assert frame.self_time_us == 0.0

    def test_total_time_no_children(self):
        frame = FlameGraphFrame(name="leaf", duration_us=100.0, self_time_us=100.0)
        assert frame.total_time_us() == 100.0

    def test_total_time_with_children(self):
        child = FlameGraphFrame(name="child", duration_us=50.0, self_time_us=50.0)
        parent = FlameGraphFrame(
            name="parent", duration_us=100.0,
            children={"child": child}, self_time_us=50.0,
        )
        assert parent.total_time_us() == 100.0

    def test_to_dict(self):
        frame = FlameGraphFrame(name="test", duration_us=42.0, self_time_us=42.0)
        d = frame.to_dict()
        assert d["name"] == "test"
        assert d["duration_us"] == 42.0
        assert d["self_time_us"] == 42.0
        assert d["children"] == {}


# ---------------------------------------------------------------------------
# FlameGraphData
# ---------------------------------------------------------------------------


class TestFlameGraphData:
    def test_creation(self):
        root = FlameGraphFrame(name="root", duration_us=0.0)
        data = FlameGraphData(
            collapsed_stacks="a;b 100",
            root=root,
            total_time_us=100.0,
            num_stacks=1,
        )
        assert data.num_stacks == 1
        assert data.total_time_us == 100.0

    def test_to_dict(self):
        root = FlameGraphFrame(name="root", duration_us=50.0)
        data = FlameGraphData(
            collapsed_stacks="a;b 50",
            root=root,
            total_time_us=50.0,
            num_stacks=1,
        )
        d = data.to_dict()
        assert "collapsed_stacks" in d
        assert d["total_time_us"] == 50.0
        assert d["num_stacks"] == 1


# ---------------------------------------------------------------------------
# _build_stack_path
# ---------------------------------------------------------------------------


class TestBuildStackPath:
    def test_gemm_kernel_with_layer(self):
        kernel = _make_kernel("sgemm_128", 100.0, "gemm", "model.encoder.linear")
        path = _build_stack_path(kernel)
        assert path.startswith("compute/matmul;")
        assert "model" in path
        assert "encoder" in path
        assert "linear" in path
        assert "sgemm_128" in path

    def test_kernel_without_layer(self):
        kernel = _make_kernel("add_kernel", 50.0, "elementwise", None)
        path = _build_stack_path(kernel)
        assert path.startswith("compute/elementwise;")
        assert "add_kernel" in path

    def test_void_prefix_stripped(self):
        kernel = _make_kernel("void my_kernel<float>()", 50.0, "other")
        path = _build_stack_path(kernel)
        assert "void " not in path
        assert "my_kernel" in path

    def test_template_args_stripped(self):
        kernel = _make_kernel("my_kernel<float, int>", 50.0, "other")
        path = _build_stack_path(kernel)
        assert "<" not in path

    def test_function_args_stripped(self):
        kernel = _make_kernel("my_kernel(float*, int)", 50.0, "other")
        path = _build_stack_path(kernel)
        assert "(" not in path

    def test_communication_kernel(self):
        kernel = _make_kernel("ncclKernel", 100.0, "communication")
        path = _build_stack_path(kernel)
        assert path.startswith("communication;")

    def test_memory_kernel(self):
        kernel = _make_kernel("memcpy_h2d", 100.0, "memory")
        path = _build_stack_path(kernel)
        assert path.startswith("memory;")

    def test_all_categories_mapped(self):
        for category in _CATEGORY_GROUP:
            kernel = _make_kernel("test", 1.0, category)
            path = _build_stack_path(kernel)
            assert len(path) > 0


# ---------------------------------------------------------------------------
# FlameGraphGenerator — generate
# ---------------------------------------------------------------------------


class TestFlameGraphGenerate:
    def test_empty_result_with_no_data(self):
        gen = FlameGraphGenerator()
        data = gen.generate()
        assert data.collapsed_stacks == ""
        assert data.num_stacks == 0
        assert data.total_time_us == 0.0

    def test_empty_result_with_empty_kernel_result(self):
        empty = KernelTraceResult(
            kernels=[], total_kernel_time_us=0.0,
            top_kernels=[], inefficient_kernels=[],
            category_breakdown={},
        )
        gen = FlameGraphGenerator(kernel_result=empty)
        data = gen.generate()
        assert data.num_stacks == 0

    def test_generate_with_kernels(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()
        assert data.num_stacks > 0
        assert data.total_time_us > 0
        assert len(data.collapsed_stacks) > 0

    def test_collapsed_stacks_format(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()
        for line in data.collapsed_stacks.strip().splitlines():
            parts = line.rsplit(" ", 1)
            assert len(parts) == 2
            assert int(parts[1]) > 0
            assert ";" in parts[0]

    def test_tree_structure(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()
        assert data.root.name == "root"
        assert data.root.duration_us > 0
        assert len(data.root.children) > 0

    def test_min_duration_filter(self):
        kernels = [
            _make_kernel("big_kernel", 1000.0, "gemm"),
            _make_kernel("tiny_kernel", 1.0, "elementwise"),
        ]
        result = _make_kernel_result(kernels)

        gen_no_filter = FlameGraphGenerator(kernel_result=result)
        data_no_filter = gen_no_filter.generate()

        gen_filtered = FlameGraphGenerator(
            kernel_result=result, min_duration_us=10.0,
        )
        data_filtered = gen_filtered.generate()

        assert data_filtered.total_time_us < data_no_filter.total_time_us

    def test_duplicate_kernels_merged(self):
        kernels = [
            _make_kernel("sgemm", 100.0, "gemm"),
            _make_kernel("sgemm", 200.0, "gemm"),
        ]
        result = _make_kernel_result(kernels)
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()
        # Both should be merged into a single stack entry.
        lines = data.collapsed_stacks.strip().splitlines()
        assert len(lines) == 1
        assert "300" in lines[0]


# ---------------------------------------------------------------------------
# FlameGraphGenerator — write_collapsed
# ---------------------------------------------------------------------------


class TestFlameGraphWriteCollapsed:
    def test_write_collapsed(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".folded", delete=False,
        ) as f:
            path = f.name

        try:
            out = gen.write_collapsed(data, path)
            assert out == path
            assert os.path.exists(path)
            content = open(path).read()
            assert len(content) > 0
            for line in content.strip().splitlines():
                parts = line.rsplit(" ", 1)
                assert len(parts) == 2
        finally:
            os.unlink(path)

    def test_write_collapsed_creates_dirs(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "profile.folded")
            gen.write_collapsed(data, path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# FlameGraphGenerator — write_svg
# ---------------------------------------------------------------------------


class TestFlameGraphWriteSVG:
    def test_write_svg(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".svg", delete=False,
        ) as f:
            path = f.name

        try:
            out = gen.write_svg(data, path)
            assert out == path
            assert os.path.exists(path)
            content = open(path).read()
            assert content.startswith("<svg")
            assert "</svg>" in content
        finally:
            os.unlink(path)

    def test_svg_with_empty_data(self):
        gen = FlameGraphGenerator()
        data = gen.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".svg", delete=False,
        ) as f:
            path = f.name

        try:
            gen.write_svg(data, path)
            content = open(path).read()
            assert "<svg" in content
            assert "No data" in content
        finally:
            os.unlink(path)

    def test_svg_custom_title(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".svg", delete=False,
        ) as f:
            path = f.name

        try:
            gen.write_svg(data, path, title="My Custom Title")
            content = open(path).read()
            assert "My Custom Title" in content
        finally:
            os.unlink(path)

    def test_svg_custom_width(self):
        result = _make_kernel_result()
        gen = FlameGraphGenerator(kernel_result=result)
        data = gen.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".svg", delete=False,
        ) as f:
            path = f.name

        try:
            gen.write_svg(data, path, width=800)
            content = open(path).read()
            assert 'width="800"' in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _svg_escape
# ---------------------------------------------------------------------------


class TestSVGEscape:
    def test_escapes_ampersand(self):
        assert _svg_escape("a&b") == "a&amp;b"

    def test_escapes_lt_gt(self):
        assert _svg_escape("<tag>") == "&lt;tag&gt;"

    def test_escapes_quotes(self):
        assert _svg_escape('"hello"') == "&quot;hello&quot;"

    def test_no_escape_needed(self):
        assert _svg_escape("simple_text") == "simple_text"
