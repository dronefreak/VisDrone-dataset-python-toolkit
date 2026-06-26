r"""Benchmark script for VisDrone detection models.

Measures and reports:
- Parameter count and model file size (MB)
- FLOPs / GFLOPs at the given input resolution
- Latency statistics: mean, median, std, P95 (ms) — GPU-synced
- Throughput: images/second
- Peak GPU memory (VRAM) allocated and reserved (MB)
- GPU utilisation % (NVIDIA only, via pynvml)
- CPU RAM used by the process (MB, via psutil)
- Model architecture summary (optional)

Works for all registered model families:
  YOLO / RT-DETR  → Ultralytics engine (model.info() for FLOPs)
  Torchvision     → fvcore FlopCountAnalysis
  RF-DETR         → thop (profile)

Usage:
  # Benchmark a pretrained checkpoint
  python scripts/benchmark.py \
      --model yolov8n \
      --checkpoint outputs/yolov8n/weights/best.pt \
      --imgsz 640 --batch-size 1 --iterations 200 --device cuda

  # Benchmark without a checkpoint (random weights, useful for architecture comparison)
  python scripts/benchmark.py --model yolov8x --imgsz 640 --no-checkpoint

  # Compare several models (pipe-friendly JSON output)
  for m in yolov8n yolov8x yolo11x; do
    python scripts/benchmark.py --model $m --no-checkpoint --json-out benchmarks/${m}.json
  done

  # Full summary with architecture table
  python scripts/benchmark.py --model yolov8n --no-checkpoint --summary
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# GPU monitoring helpers
# ---------------------------------------------------------------------------


def _nvml_handle(device_index: int = 0):
    """Return an NVML device handle for GPU core-utilisation %, or None."""
    try:
        import pynvml  # nvidia-ml-py (or legacy pynvml)

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        return None


def _gpu_core_util_pct(handle) -> float:
    """Return GPU core utilisation % via pynvml, or NaN if unavailable."""
    if handle is None:
        return float("nan")
    try:
        import pynvml

        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return float("nan")


def _vram_info(device: torch.device) -> dict[str, float]:
    """Return VRAM stats using torch.cuda (no pynvml dependency).

    Uses ``torch.cuda.mem_get_info()`` for total/free and
    ``torch.cuda.max_memory_allocated()`` for the peak used during the run.
    """
    if device.type != "cuda":
        return {
            "vram_peak_mb": float("nan"),
            "vram_reserved_mb": float("nan"),
            "vram_total_mb": float("nan"),
            "vram_util_pct": float("nan"),
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    total_mb = total_bytes / 1024**2
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2

    return {
        "vram_peak_mb": peak_alloc_mb,
        "vram_reserved_mb": peak_reserved_mb,
        "vram_total_mb": total_mb,
        "vram_util_pct": peak_alloc_mb / total_mb * 100 if total_mb > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# FLOPs calculation — per model family
# ---------------------------------------------------------------------------


def _flops_ultralytics(model_wrapper, imgsz: int) -> tuple[float, int]:
    """Return (GFLOPs, param_count) via Ultralytics model.info()."""
    try:
        import contextlib
        import io
        import logging

        buf = io.StringIO()
        # Silence both stdout and ultralytics logger during the call
        ul_logger = logging.getLogger("ultralytics")
        prev_level = ul_logger.level
        ul_logger.setLevel(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(buf):
                info = model_wrapper.info(verbose=True, imgsz=imgsz)
        finally:
            ul_logger.setLevel(prev_level)

        # info = (layers, params, gradients, GFLOPs)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            return float(info[3]), int(info[1])
        if isinstance(info, dict):
            return float(info.get("GFLOPs", float("nan"))), int(info.get("parameters", 0))
    except Exception:
        pass
    # Fallback: count params manually
    params = sum(p.numel() for p in model_wrapper.model.parameters())
    return float("nan"), params


def _flops_fvcore(model: torch.nn.Module, imgsz: int, batch_size: int) -> tuple[float, int]:
    """Return (GFLOPs, param_count) via fvcore FlopCountAnalysis."""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count

        dummy = torch.zeros(batch_size, 3, imgsz, imgsz).to(next(model.parameters()).device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_settings(raise_if_no_op=False)
            gflops = flops.total() / 1e9
        params = parameter_count(model)[""]
        return float(gflops), int(params)
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return float("nan"), params


def _flops_thop(model: torch.nn.Module, imgsz: int, batch_size: int) -> tuple[float, int]:
    """Return (GFLOPs, param_count) via thop."""
    try:
        from thop import profile

        dummy = torch.zeros(batch_size, 3, imgsz, imgsz).to(next(model.parameters()).device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, params = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9, int(params)  # MACs × 2 = FLOPs
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return float("nan"), params


# ---------------------------------------------------------------------------
# Model size on disk
# ---------------------------------------------------------------------------


def _model_size_mb(checkpoint_path: str | None) -> float:
    """Return checkpoint file size in MB, or NaN if not given."""
    if checkpoint_path and Path(checkpoint_path).exists():
        return Path(checkpoint_path).stat().st_size / 1024**2
    return float("nan")


# ---------------------------------------------------------------------------
# Architecture summary
# ---------------------------------------------------------------------------


def _print_summary(model: torch.nn.Module, imgsz: int, batch_size: int) -> None:
    """Print torchinfo model summary if available."""
    try:
        from torchinfo import summary

        summary(
            model,
            input_size=(batch_size, 3, imgsz, imgsz),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=3,
            verbose=1,
        )
    except Exception as exc:
        console.print(f"[dim]torchinfo summary unavailable: {exc}[/dim]")


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def _measure_latency(
    forward_fn,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float]:
    """Time `forward_fn()` for ``iterations`` runs, returning latency stats (ms)."""
    use_cuda = device.type == "cuda"

    if use_cuda:
        start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_evt = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            forward_fn()
    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    latencies_ms: list[float] = []
    for i in range(iterations):
        if use_cuda:
            start_evt[i].record()
            with torch.no_grad():
                forward_fn()
            end_evt[i].record()
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                forward_fn()
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    if use_cuda:
        torch.cuda.synchronize()
        latencies_ms = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Per-family benchmarking
# ---------------------------------------------------------------------------


def _benchmark_ultralytics(args, handle) -> dict[str, Any]:
    """Benchmark a YOLO or RT-DETR model."""
    is_rtdetr = args.model.lower().startswith("rtdetr")
    if is_rtdetr:
        from ultralytics import RTDETR

        ul_model = RTDETR(args.checkpoint or f"{args.model}.pt")
    else:
        from ultralytics import YOLO

        ul_model = YOLO(args.checkpoint or f"{args.model}.pt")

    device = torch.device(args.device)
    ul_model.to(device)

    gflops, params = _flops_ultralytics(ul_model, args.imgsz)

    dummy = torch.zeros(args.batch_size, 3, args.imgsz, args.imgsz).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _fwd():
        return ul_model.model(dummy)

    latency = _measure_latency(_fwd, device, warmup=args.warmup, iterations=args.iterations)

    vram = _vram_info(device)
    process = psutil.Process(os.getpid())

    return {
        "model": args.model,
        "family": "RT-DETR (ultralytics)" if is_rtdetr else "YOLO (ultralytics)",
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "params_M": params / 1e6,
        "gflops": gflops,
        "file_size_mb": _model_size_mb(args.checkpoint),
        **latency,
        "fps": args.batch_size / (latency["mean_ms"] / 1000),
        **vram,
        "gpu_core_util_pct": _gpu_core_util_pct(handle),
        "cpu_ram_mb": process.memory_info().rss / 1024**2,
    }


def _benchmark_torchvision(args, handle) -> dict[str, Any]:
    """Benchmark a torchvision model (FasterRCNN, FCOS, RetinaNet)."""
    from visdrone_toolkit.utils import get_model

    device = torch.device(args.device)

    console.print("[dim]Loading torchvision model...[/dim]")
    model = get_model(
        args.model, num_classes=12, pretrained=not args.no_checkpoint, device=str(device)
    )

    if args.checkpoint and not args.no_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    tv_model = model.model if hasattr(model, "model") else model
    tv_model.eval().to(device)

    gflops, params = _flops_fvcore(tv_model, args.imgsz, args.batch_size)

    dummy = [torch.zeros(3, args.imgsz, args.imgsz).to(device) for _ in range(args.batch_size)]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _fwd():
        return tv_model(dummy)

    latency = _measure_latency(_fwd, device, warmup=args.warmup, iterations=args.iterations)

    if args.summary:
        _print_summary(tv_model, args.imgsz, args.batch_size)

    vram = _vram_info(device)
    process = psutil.Process(os.getpid())

    return {
        "model": args.model,
        "family": "torchvision",
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "params_M": params / 1e6,
        "gflops": gflops,
        "file_size_mb": _model_size_mb(args.checkpoint),
        **latency,
        "fps": args.batch_size / (latency["mean_ms"] / 1000),
        **vram,
        "gpu_core_util_pct": _gpu_core_util_pct(handle),
        "cpu_ram_mb": process.memory_info().rss / 1024**2,
    }


def _benchmark_rfdetr(args, handle) -> dict[str, Any]:
    """Benchmark an RF-DETR model."""
    try:
        import rfdetr as _rfdetr_pkg
    except ImportError as err:
        raise ImportError("pip install rfdetr") from err

    from visdrone_toolkit.rfdetr_trainer import _MODEL_CLASS_MAP, _RFDETR_CLASSES

    cls_name = _MODEL_CLASS_MAP.get(args.model, "RFDETRLarge")
    model_cls = getattr(_rfdetr_pkg, cls_name)

    device = torch.device(args.device)
    console.print("[dim]Loading RF-DETR model...[/dim]")

    kwargs: dict[str, Any] = {"num_classes": len(_RFDETR_CLASSES), "device": str(device)}
    if args.checkpoint and not args.no_checkpoint:
        kwargs["pretrain_weights"] = args.checkpoint
    rf_model = model_cls(**kwargs)

    inner: torch.nn.Module = rf_model.model
    inner.eval()

    gflops, params = _flops_thop(inner, args.imgsz, args.batch_size)

    dummy = torch.zeros(args.batch_size, 3, args.imgsz, args.imgsz).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _fwd():
        return inner(dummy)

    latency = _measure_latency(_fwd, device, warmup=args.warmup, iterations=args.iterations)

    if args.summary:
        _print_summary(inner, args.imgsz, args.batch_size)

    vram = _vram_info(device)
    process = psutil.Process(os.getpid())

    return {
        "model": args.model,
        "family": "RF-DETR (rfdetr)",
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "params_M": params / 1e6,
        "gflops": gflops,
        "file_size_mb": _model_size_mb(args.checkpoint),
        **latency,
        "fps": args.batch_size / (latency["mean_ms"] / 1000),
        **vram,
        "gpu_core_util_pct": _gpu_core_util_pct(handle),
        "cpu_ram_mb": process.memory_info().rss / 1024**2,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "[dim]N/A[/dim]"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def print_benchmark_table(result: dict[str, Any]) -> None:
    """Print a rich benchmark results table."""
    console.rule(f"[bold]Benchmark — {result['model']}[/bold]")
    console.print(f"  Family : [cyan]{result['family']}[/cyan]")
    console.print(f"  Image  : {result['imgsz']}×{result['imgsz']}  Batch: {result['batch_size']}")

    # --- Architecture ---
    arch_table = Table(
        title="Architecture", show_header=True, header_style="bold magenta", min_width=50
    )
    arch_table.add_column("Metric", style="cyan")
    arch_table.add_column("Value", justify="right")
    arch_table.add_row("Parameters (M)", _fmt(result.get("params_M"), 2))
    arch_table.add_row("GFLOPs", _fmt(result.get("gflops"), 2))
    arch_table.add_row("Checkpoint size (MB)", _fmt(result.get("file_size_mb"), 1))
    console.print(arch_table)

    # --- Latency ---
    lat_table = Table(
        title="Latency (ms)", show_header=True, header_style="bold magenta", min_width=50
    )
    lat_table.add_column("Metric", style="cyan")
    lat_table.add_column("Value", justify="right")
    lat_table.add_row("Mean", _fmt(result.get("mean_ms"), 2))
    lat_table.add_row("Median", _fmt(result.get("median_ms"), 2))
    lat_table.add_row("Std", _fmt(result.get("std_ms"), 2))
    lat_table.add_row("P95", _fmt(result.get("p95_ms"), 2))
    lat_table.add_row("Min", _fmt(result.get("min_ms"), 2))
    lat_table.add_row("Max", _fmt(result.get("max_ms"), 2))
    lat_table.add_row("[bold]Throughput (FPS)[/bold]", f"[bold]{_fmt(result.get('fps'), 1)}[/bold]")
    console.print(lat_table)

    # --- Memory ---
    mem_table = Table(title="Memory", show_header=True, header_style="bold magenta", min_width=50)
    mem_table.add_column("Metric", style="cyan")
    mem_table.add_column("Value", justify="right")
    mem_table.add_row("VRAM total (MB)", _fmt(result.get("vram_total_mb"), 0))
    mem_table.add_row("VRAM peak allocated (MB)", _fmt(result.get("vram_peak_mb"), 1))
    mem_table.add_row("VRAM peak reserved (MB)", _fmt(result.get("vram_reserved_mb"), 1))

    vram_util = result.get("vram_util_pct", float("nan"))
    if isinstance(vram_util, float) and not np.isnan(vram_util):
        mem_table.add_row("VRAM utilisation %", f"{vram_util:.1f}%")
    else:
        mem_table.add_row("VRAM utilisation %", "[dim]N/A[/dim]")

    gpu_core = result.get("gpu_core_util_pct", float("nan"))
    if isinstance(gpu_core, float) and not np.isnan(gpu_core):
        mem_table.add_row("GPU core utilisation %", f"{gpu_core:.1f}%")
    else:
        mem_table.add_row("GPU core utilisation %", "[dim]N/A — run nvidia-smi separately[/dim]")

    mem_table.add_row("Process RAM (MB)", _fmt(result.get("cpu_ram_mb"), 1))
    console.print(mem_table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VisDrone detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--model", required=True, help="Model name (e.g. yolov8n, rfdetr-large)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.pt / .pth). Omit together with --no-checkpoint for pretrained weights.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip loading a checkpoint (use downloaded pretrained weights for YOLO, random for others).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square)")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for latency measurement"
    )
    parser.add_argument("--iterations", type=int, default=200, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda, cpu, cuda:0, ...)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print torchinfo layer-by-layer summary"
    )
    parser.add_argument("--json-out", default=None, help="Write JSON results to this file path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint and not args.no_checkpoint:
        console.print(
            "[yellow]No --checkpoint provided. Using pretrained / random weights. "
            "Pass --no-checkpoint to suppress this message.[/yellow]"
        )

    console.print("\n[bold cyan]VisDrone Model Benchmark[/bold cyan]")
    console.print(f"  Model     : [bold]{args.model}[/bold]")
    console.print(f"  Device    : {args.device}")
    console.print(f"  Image size: {args.imgsz}×{args.imgsz}   Batch: {args.batch_size}")
    console.print(f"  Iterations: {args.iterations}  Warmup: {args.warmup}\n")

    # Initialise GPU monitor (best-effort)
    device_idx = 0
    if ":" in args.device:
        try:
            device_idx = int(args.device.split(":")[1])
        except ValueError:
            pass
    handle = _nvml_handle(device_idx)

    name = args.model.lower()
    if name.startswith(("yolo", "rtdetr")):
        result = _benchmark_ultralytics(args, handle)
    elif name.startswith("rfdetr"):
        result = _benchmark_rfdetr(args, handle)
    else:
        result = _benchmark_torchvision(args, handle)

    print_benchmark_table(result)

    # JSON export
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            return v

        serialisable = {k: _serialise(v) for k, v in result.items()}
        out_path.write_text(json.dumps(serialisable, indent=2))
        console.print(f"\n[green]✓[/green] Results saved to [bold]{args.json_out}[/bold]")


if __name__ == "__main__":
    main()
