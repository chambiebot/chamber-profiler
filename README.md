# chamber-profiler

One command to profile any ML training job. Zero config. Beautiful output.

## Install

```bash
pip install chamber-profiler

# With GPU support
pip install chamber-profiler[all]

# Development
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Profile a training script
chamber-profile run python train.py --epochs 10

# Save results to JSON
chamber-profile run --output profile.json python train.py

# Generate HTML report
chamber-profile run --html report.html python train.py

# Analyze a saved profile
chamber-profile analyze profile.json

# Generate a report from a saved profile
chamber-profile report profile.json --format html --output report.html

# Compare two profiles
chamber-profile compare baseline.json optimized.json

# Live GPU dashboard
chamber-profile live

# Upload to Chamber platform
chamber-profile upload profile.json --api-key $CHAMBER_API_KEY
```

### Python SDK

```python
from src.sdk import ChamberProfiler

# Context manager
profiler = ChamberProfiler()
with profiler.profile():
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

profiler.report()  # terminal
profiler.report(format="html", output_path="report.html")

# Manual start/stop
profiler = ChamberProfiler()
profiler.start()
train()
summary = profiler.stop()
print(summary.overall_efficiency, summary.primary_bottleneck)
```

### Decorators

```python
from src.sdk import profile_training

@profile_training
def train():
    ...

@profile_training(report_format="html", output_path="report.html")
def train():
    ...
```

## What It Profiles

| Module | Metrics |
|--------|---------|
| **GPU Profiler** | Utilization, SM activity, memory bandwidth, power, temperature, clock speed |
| **Kernel Tracer** | CUDA kernel execution times, categories (GEMM, conv, attention, etc.), occupancy estimates |
| **Memory Analyzer** | Allocation tracking, leak detection, OOM risk assessment, memory breakdown |
| **Communication Profiler** | Collective op timing (allreduce, allgather, etc.), bandwidth, straggler detection |
| **Data Loading Profiler** | Batch load time, GPU idle time, I/O throughput, DataLoader config analysis |

## Bottleneck Detection

The bottleneck detector automatically identifies performance issues and provides actionable recommendations:

- Low GPU/SM utilization
- Memory-bound kernels
- Excessive memory copies
- Memory leaks and OOM risk
- Communication overhead in distributed training
- Data loading bottlenecks
- Unexplained GPU idle time

Each bottleneck includes an impact score, description, specific recommendations, and expected speedup estimate.

## Report Formats

- **Terminal**: Rich-formatted tables with color-coded metrics
- **HTML**: Self-contained interactive report with charts
- **JSON**: Structured data for programmatic consumption

## Chamber Integration

Upload profiles to the [Chamber platform](https://app.usechamber.io) for team-wide visibility:

```bash
export CHAMBER_API_KEY=your_key
chamber-profile run --upload python train.py
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

## License

Apache-2.0
