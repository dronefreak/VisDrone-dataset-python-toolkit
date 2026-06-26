r"""Convert VisDrone model checkpoints to safetensors format for HuggingFace upload.

Supports:
  - Torchvision models (.pt with model_state_dict key)
  - Ultralytics YOLO models (.pt Ultralytics format)

Output: a .safetensors file plus a metadata.json sidecar.

Usage:
  # Torchvision
  python scripts/convert_to_safetensors.py \\
      --checkpoint outputs/fasterrcnn_200ep/best.pt \\
      --model fasterrcnn_resnet50 \\
      --output-dir hf_upload/fasterrcnn_resnet50

  # YOLO
  python scripts/convert_to_safetensors.py \\
      --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \\
      --model yolov8n \\
      --output-dir hf_upload/yolov8n
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

_YOLO_ARCHITECTURES = {
    "yolov8": "Ultralytics YOLOv8",
    "yolov9": "Ultralytics YOLOv9",
    "yolov10": "Ultralytics YOLOv10",
    "yolo11": "Ultralytics YOLO11",
    "yolo26": "Ultralytics YOLO26",
}

_TORCHVISION_ARCHITECTURES = {
    "fasterrcnn_resnet50": "Faster R-CNN ResNet50 FPN",
    "fasterrcnn_mobilenet": "Faster R-CNN MobileNetV3 Large FPN",
    "fcos_resnet50": "FCOS ResNet50 FPN",
    "retinanet_resnet50": "RetinaNet ResNet50 FPN",
}


def _is_yolo(model_name: str) -> bool:
    return model_name.lower().startswith("yolo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone checkpoint → safetensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model", required=True, help="Model name (e.g. yolov8n, fasterrcnn_resnet50)"
    )
    parser.add_argument("--output-dir", default="hf_upload", help="Output directory")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of VisDrone classes (default 11, ignoring 'ignored-regions')",
    )
    parser.add_argument(
        "--extra-meta",
        nargs="*",
        metavar="KEY=VALUE",
        help="Extra metadata pairs, e.g. --extra-meta epochs=200 f1=0.667",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# State-dict extraction
# ---------------------------------------------------------------------------


def _load_torchvision_state_dict(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a torchvision checkpoint and return (state_dict, training_meta)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {
            k: v
            for k, v in ckpt.items()
            if k not in ("model_state_dict", "optimizer_state_dict")
            and not isinstance(v, (torch.Tensor, dict))
        }
    elif isinstance(ckpt, dict):
        # Raw state dict saved directly
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError(
            f"Unrecognised torchvision checkpoint format in {checkpoint_path}.\n"
            "Expected a dict with 'model_state_dict' key."
        )

    # Verify all values are tensors
    bad = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
    if bad:
        raise ValueError(f"State dict contains non-tensor values for keys: {bad[:5]}")

    return state_dict, meta


def _load_yolo_state_dict(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load an Ultralytics YOLO checkpoint and return (state_dict, training_meta)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unrecognised YOLO checkpoint format in {checkpoint_path}.\n"
            f"Expected a dict, got {type(ckpt).__name__}."
        )

    # Prefer EMA weights (more accurate) over the last-epoch weights
    model_obj = ckpt.get("ema") or ckpt.get("model")
    if model_obj is None:
        raise ValueError(
            f"Could not find 'model' or 'ema' key in {checkpoint_path}.\n"
            f"Available keys: {list(ckpt.keys())}"
        )

    # model_obj may be wrapped; unwrap to nn.Module if needed
    if hasattr(model_obj, "module"):
        model_obj = model_obj.module

    state_dict = model_obj.float().state_dict()

    meta = {}
    for k in ("epoch", "best_fitness", "date", "version"):
        if k in ckpt and ckpt[k] is not None:
            meta[k] = ckpt[k]
    # Extract training args subset
    if "train_args" in ckpt and isinstance(ckpt["train_args"], dict):
        wanted = ("imgsz", "batch", "lr0", "epochs", "device", "amp")
        for k in wanted:
            if k in ckpt["train_args"]:
                meta[f"train_{k}"] = ckpt["train_args"][k]

    return state_dict, meta


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    checkpoint_path: Path,
    model_name: str,
    output_dir: Path,
    num_classes: int,
    extra_meta: dict[str, str],
) -> None:
    from safetensors.torch import save_file  # noqa: PLC0415 (lazy import)

    print(f"Loading checkpoint: {checkpoint_path}")

    if _is_yolo(model_name):
        state_dict, training_meta = _load_yolo_state_dict(checkpoint_path)
        arch_family = next(
            (v for k, v in _YOLO_ARCHITECTURES.items() if model_name.lower().startswith(k)),
            f"Ultralytics YOLO ({model_name})",
        )
    else:
        state_dict, training_meta = _load_torchvision_state_dict(checkpoint_path)
        arch_family = _TORCHVISION_ARCHITECTURES.get(model_name, model_name)

    print(f"  Architecture : {arch_family}")
    print(f"  Tensors      : {len(state_dict)}")
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"  Parameters   : {total_params:,}")

    # Safetensors requires all tensors to be contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # Build metadata (all values must be strings)
    meta: dict[str, str] = {
        "model_name": model_name,
        "architecture": arch_family,
        "dataset": "VisDrone2019-DET",
        "num_classes": str(num_classes),
        "class_names": ",".join(_VISDRONE_CLASSES[:num_classes]),
        "total_params": str(total_params),
        "source_file": checkpoint_path.name,
        "framework": "pytorch",
        "task": "object-detection",
    }
    for k, v in training_meta.items():
        meta[str(k)] = str(v)
    meta.update(extra_meta)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    sf_path = output_dir / f"{model_name}.safetensors"
    meta_path = output_dir / "metadata.json"

    save_file(state_dict, str(sf_path), metadata=meta)
    print(f"\n✓ Saved: {sf_path}  ({sf_path.stat().st_size / 1e6:.1f} MB)")

    # Write a richer sidecar JSON (safetensors metadata values are limited to strings)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "architecture": arch_family,
                "dataset": "VisDrone2019-DET",
                "num_classes": num_classes,
                "class_names": _VISDRONE_CLASSES[:num_classes],
                "total_params": total_params,
                "training": training_meta,
                "extra": extra_meta,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"✓ Saved: {meta_path}")

    # Smoke-test: re-load and verify tensor count
    _verify(sf_path, len(state_dict))


def _verify(sf_path: Path, expected_count: int) -> None:
    """Reload the safetensors file and assert tensor count matches."""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    if len(keys) != expected_count:
        print(
            f"  [WARN] Expected {expected_count} tensors, found {len(keys)} after reload.",
            file=sys.stderr,
        )
    else:
        print(f"✓ Verified: {len(keys)} tensors round-trip OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Parse --extra-meta KEY=VALUE pairs
    extra_meta: dict[str, str] = {}
    for pair in args.extra_meta or []:
        if "=" not in pair:
            print(f"Warning: ignoring malformed --extra-meta entry {pair!r} (no '=')")
            continue
        k, _, v = pair.partition("=")
        extra_meta[k.strip()] = v.strip()

    convert(
        checkpoint_path=checkpoint_path,
        model_name=args.model,
        output_dir=Path(args.output_dir),
        num_classes=args.num_classes,
        extra_meta=extra_meta,
    )

    print(f"\nOutput ready in: {args.output_dir}/")
    print("Next step: see docs/HF_UPLOAD_GUIDE.md")


if __name__ == "__main__":
    main()
