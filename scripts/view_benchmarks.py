#!/usr/bin/env python3
"""Display benchmark results for all models in a formatted table."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_benchmarks(benchmark_dir: str = "benchmarks") -> dict:
    """Load all benchmark JSON files."""
    benchmarks = {}
    for file_path in Path(benchmark_dir).glob("*.json"):
        with open(file_path) as f:
            data = json.load(f)
            benchmarks[data.get("model_name", file_path.stem)] = data
    return benchmarks


def print_benchmark_table(benchmarks: dict) -> None:
    """Print benchmarks in a rich table."""
    table = Table(title="Model Benchmarks", header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("FLOPs", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("mAP@0.5", justify="right")
    table.add_column("mAP@0.5:0.95", justify="right")
    table.add_column("Inference (ms)", justify="right")

    for name, data in sorted(benchmarks.items()):
        table.add_row(
            name,
            data.get("parameters", "N/A"),
            data.get("flops", "N/A"),
            f"{data.get('inference', {}).get('fps', 0):.1f}",
            f"{data.get('evaluation', {}).get('mAP50', 0):.3f}",
            f"{data.get('evaluation', {}).get('mAP50_95', 0):.3f}",
            f"{data.get('inference', {}).get('avg_ms', 0):.1f}",
        )

    console.print(table)


def main() -> None:
    """Main entry point for benchmark viewer."""
    benchmarks = load_benchmarks()
    if not benchmarks:
        console.print("[red]No benchmark files found in 'benchmarks/' directory.[/red]")
        return
    print_benchmark_table(benchmarks)


if __name__ == "__main__":
    main()
