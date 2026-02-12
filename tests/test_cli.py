"""Tests for src.cli.main."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from src.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_profile():
    """Create a temporary sample profile JSON."""
    data = {
        "metadata": {
            "tool": "chamber-profiler",
            "version": "0.1.0",
            "timestamp": "2025-01-01T00:00:00",
            "command": "python train.py",
        },
        "summary": {
            "total_time_s": 10.0,
            "compute_pct": 70.0,
            "memory_pct": 10.0,
            "communication_pct": 10.0,
            "data_loading_pct": 5.0,
            "idle_pct": 5.0,
            "primary_bottleneck": "compute",
            "bottlenecks": [],
            "overall_efficiency": 0.7,
            "summary_text": "Test summary.",
        },
        "gpu_profile": {
            "duration_s": 10.0,
            "gpu_count": 1,
            "avg_utilization": 70.0,
            "peak_memory_mb": 4000.0,
            "avg_power_watts": 200.0,
            "snapshots": [],
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


class TestCLIGroup:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Chamber Profiler" in result.output


class TestRunCommand:
    def test_run_no_command(self, runner):
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    def test_run_command_not_found(self, runner):
        result = runner.invoke(cli, ["run", "nonexistent_command_xyz_12345"])
        assert result.exit_code != 0

    def test_run_simple_command(self, runner):
        result = runner.invoke(cli, ["run", "echo", "hello"])
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_run_with_output(self, runner):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = runner.invoke(
                cli, ["run", "--output", path, "echo", "test"],
            )
            assert result.exit_code == 0
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "metadata" in data
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestAnalyzeCommand:
    def test_analyze(self, runner, sample_profile):
        result = runner.invoke(cli, ["analyze", sample_profile])
        assert result.exit_code == 0

    def test_analyze_missing_file(self, runner):
        result = runner.invoke(cli, ["analyze", "/nonexistent/path.json"])
        assert result.exit_code != 0


class TestReportCommand:
    def test_report_terminal(self, runner, sample_profile):
        result = runner.invoke(
            cli, ["report", sample_profile, "--format", "terminal"],
        )
        assert result.exit_code == 0

    def test_report_json(self, runner, sample_profile):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = runner.invoke(
                cli, [
                    "report", sample_profile,
                    "--format", "json",
                    "--output", out_path,
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_report_html(self, runner, sample_profile):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            result = runner.invoke(
                cli, [
                    "report", sample_profile,
                    "--format", "html",
                    "--output", out_path,
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestCompareCommand:
    def test_compare(self, runner, sample_profile):
        result = runner.invoke(
            cli, ["compare", sample_profile, sample_profile],
        )
        assert result.exit_code == 0

    def test_compare_missing_file(self, runner, sample_profile):
        result = runner.invoke(
            cli, ["compare", sample_profile, "/nonexistent.json"],
        )
        assert result.exit_code != 0


class TestUploadCommand:
    def test_upload_no_api_key(self, runner, sample_profile):
        """Upload should fail without API key."""
        result = runner.invoke(
            cli, ["upload", sample_profile],
            env={"CHAMBER_API_KEY": ""},
        )
        # Should fail or warn about missing API key
        assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()
