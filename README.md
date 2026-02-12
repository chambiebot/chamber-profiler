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
| **Distributed Profiler** | NCCL allreduce time, gradient sync overhead, communication vs compute ratio |
| **Ray Profiler** | Ray Train actor metrics, actor-to-GPU mapping, straggler detection across workers |
| **Cost Analyzer** | $/hour cost estimation, cheaper GPU recommendations, monthly savings projections |
| **Timeline Generator** | Chrome trace format JSON for visualization in chrome://tracing or Perfetto UI |

## Distributed Training Profiling

Profile multi-node distributed training to identify communication bottlenecks:

```python
from src.profiler.distributed_profiler import DistributedProfiler

profiler = DistributedProfiler()
profiler.start()
for step in range(num_steps):
    loss = model(batch)
    loss.backward()
    profiler.record_step(step, compute_time_us, sync_time_us)
    optimizer.step()
result = profiler.stop()
print(result.comm_compute_ratio, result.gradient_sync_overhead_pct)
for rec in result.recommendations:
    print(rec)
```

## Ray Train Profiling

Profile Ray Train jobs — detect stragglers, map actors to GPUs:

```python
from src.profiler.ray_profiler import RayProfiler

profiler = RayProfiler()
profiler.start()
# ... Ray Train job runs ...
result = profiler.stop()
for straggler in result.stragglers:
    print(straggler.actor_id, straggler.slowdown_factor)
```

## GPU Cost Analysis

Estimate costs and find cheaper GPU configurations:

```python
from src.analysis.cost_analyzer import CostAnalyzer

analyzer = CostAnalyzer(gpu_name="a100_80gb", num_gpus=4)
result = analyzer.analyze()
print(result.current_estimate.total_cost_per_hour)
for rec in result.recommendations:
    print(rec.gpu_name, f"saves {rec.estimated_savings_pct:.0f}%")
print(f"Potential monthly savings: ${result.potential_monthly_savings:,.0f}")
```

## Timeline Visualization

Generate Chrome trace format timelines viewable in chrome://tracing or Perfetto UI:

```python
from src.profiler.timeline_generator import TimelineGenerator

generator = TimelineGenerator(
    gpu_result=gpu_result,
    kernel_result=kernel_result,
    comm_result=comm_result,
    data_result=data_result,
)
generator.generate("timeline.json")
# Open chrome://tracing and load timeline.json
```

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
