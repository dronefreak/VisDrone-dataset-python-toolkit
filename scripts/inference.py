r"""Inference script for VisDrone object detection models.

Supports inference on:
- Single images
- Directories of images
- Video files
- All registered models (torchvision, YOLO, DETR)
- Soft-NMS post-processing

Usage examples:
  # Image directory, YOLO model
  python scripts/inference.py \\
      --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \\
      --model yolov8n --input data/images/

  # Single image, torchvision model
  python scripts/inference.py \\
      --checkpoint outputs/fasterrcnn/best.pt \\
      --model fasterrcnn_resnet50 --input data/images/frame.jpg

  # Video file
  python scripts/inference.py \\
      --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \\
      --model yolov8n --input video.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on VisDrone models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint / .pt file")
    parser.add_argument("--model", default="fasterrcnn_resnet50", help="Model name")
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (YOLO only)")

    # Input  (images / directory / video file)
    parser.add_argument("--input", required=True, help="Input image, directory, or video file")
    parser.add_argument("--output-dir", default="inference_outputs", help="Output directory")

    # Inference parameters
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )

    # Post-processing
    parser.add_argument("--soft-nms", action="store_true", help="Use Soft-NMS (torchvision only)")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")

    # Visualization
    parser.add_argument("--no-save-viz", action="store_true", help="Don't save visualizations")
    parser.add_argument("--show", action="store_true", help="Display results interactively")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO inference path
# ---------------------------------------------------------------------------


def run_yolo(
    checkpoint_path: str,
    input_path: Path,
    output_dir: Path,
    score_threshold: float,
    device: str,
    imgsz: int,
    show: bool,
) -> None:
    """Run YOLO inference with custom visualization."""
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    output_dir.mkdir(parents=True, exist_ok=True)

    model = UltralyticsYOLO(str(checkpoint_path))

    print(f"Running YOLO inference on {input_path} ...")

    results = model.predict(
        source=str(input_path),
        conf=score_threshold,
        device=device,
        imgsz=imgsz,
        save=False,
        verbose=True,
    )

    total_det = 0

    for result in results:
        total_det += len(result.boxes)

        # Original image (full resolution)
        frame = result.orig_img.copy()

        # Extract predictions
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        # Custom visualization
        viz = draw_detections(
            frame,
            boxes,
            scores,
            labels,
            VISDRONE_CLASSES,
        )

        # Save
        image_path = Path(result.path)
        out_path = output_dir / f"{image_path.stem}_pred.jpg"

        cv2.imwrite(str(out_path), viz)

        if show:
            cv2.imshow("YOLO Inference", viz)
            if cv2.waitKey(0) == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    print(f"\n Processed {len(results)} image(s)")
    print(f"Total detections: {total_det}")
    print(f"Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Torchvision inference path
# ---------------------------------------------------------------------------


def load_torchvision_model(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load torchvision model from checkpoint."""
    print(f"Loading {model_name} from {checkpoint_path} ...")

    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"  Loaded from epoch {checkpoint['epoch']}")
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("✓ Model loaded")
    return model


def process_image_for_torchvision(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR numpy frame to a [C, H, W] float32 tensor in [0, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def infer_torchvision_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    score_threshold: float,
    use_soft_nms: bool,
    nms_threshold: float,
) -> dict[str, np.ndarray]:
    """Run inference on a single BGR frame."""
    img_tensor = process_image_for_torchvision(frame_bgr).to(device)
    pred = model([img_tensor])[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if use_soft_nms and len(boxes) > 0:
        boxes, scores, labels = _apply_soft_nms(
            boxes,
            scores,
            labels,
            sigma=0.5,
            score_threshold=score_threshold,
            iou_threshold=nms_threshold,
        )

    return {"boxes": boxes, "scores": scores, "labels": labels}


def _apply_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class Gaussian Soft-NMS."""
    from visdrone_toolkit.soft_nms_utils import apply_soft_nms_per_class

    bt = torch.from_numpy(boxes).float()
    st = torch.from_numpy(scores).float()
    lt = torch.from_numpy(labels.astype(np.int64))
    bt, lt, st = apply_soft_nms_per_class(
        bt, lt, st, iou_threshold=iou_threshold, sigma=sigma, score_threshold=score_threshold
    )
    return bt.numpy(), st.numpy(), lt.numpy()


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    print(f"Drawing {len(boxes)} detections on frame of size {w}x{h} ...")

    # Much more conservative scaling
    scale = max(h, w) / 2000.0

    box_thickness = max(1, int(scale))
    font_scale = max(0.3, scale * 0.35)
    font_thickness = 1

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)

        # Draw box
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            box_thickness,
        )

        name = class_names[label] if label < len(class_names) else f"cls{label}"

        text = f"{name} {score:.2f}"

        # Compute text size
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        # Filled label background
        cv2.rectangle(
            out,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            (0, 255, 0),
            -1,
        )

        # Text
        cv2.putText(
            out,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return out


def run_torchvision_images(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
    output_dir: Path,
    score_threshold: float,
    use_soft_nms: bool,
    nms_threshold: float,
    save_viz: bool,
    show: bool,
) -> None:
    """Run inference on a list of image paths."""
    t0 = time.time()
    total_det = 0
    if save_viz:
        output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [warn] Could not read {image_path.name}, skipping")
            continue

        result = infer_torchvision_frame(
            model, frame, device, score_threshold, use_soft_nms, nms_threshold
        )
        total_det += len(result["boxes"])
        print(f"  {image_path.name}: {len(result['boxes'])} detections")

        if save_viz:
            viz = draw_detections(
                frame, result["boxes"], result["scores"], result["labels"], VISDRONE_CLASSES
            )
            out_path = output_dir / f"{image_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), viz)

        if show:
            viz = draw_detections(
                frame, result["boxes"], result["scores"], result["labels"], VISDRONE_CLASSES
            )
            cv2.imshow("VisDrone Inference", viz)
            if cv2.waitKey(0) == ord("q"):
                cv2.destroyAllWindows()
                break

    elapsed = time.time() - t0
    n = len(image_paths)
    print(f"\n✓ {n} images in {elapsed:.2f}s ({n / elapsed:.1f} FPS)")
    print(f"  Total detections: {total_det}")
    print(f"  Results saved to: {output_dir}")


def run_torchvision_video(
    model: torch.nn.Module,
    video_path: Path,
    device: torch.device,
    output_dir: Path,
    score_threshold: float,
    use_soft_nms: bool,
    nms_threshold: float,
    save_viz: bool,
    show: bool,
) -> None:
    """Run inference on a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer: cv2.VideoWriter | None = None
    if save_viz:
        out_path = output_dir / f"{video_path.stem}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    t0 = time.time()
    frame_idx = 0
    total_det = 0

    print(f"Processing video: {video_path.name} ({total_frames} frames @ {fps:.1f} FPS) ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = infer_torchvision_frame(
            model, frame, device, score_threshold, use_soft_nms, nms_threshold
        )
        total_det += len(result["boxes"])

        viz = draw_detections(
            frame, result["boxes"], result["scores"], result["labels"], VISDRONE_CLASSES
        )

        if writer is not None:
            writer.write(viz)

        if show:
            cv2.imshow("VisDrone Inference", viz)
            if cv2.waitKey(1) == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total_frames} — {frame_idx / elapsed:.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n✓ {frame_idx} frames in {elapsed:.2f}s ({frame_idx / elapsed:.1f} FPS)")
    print(f"  Total detections: {total_det}")
    if save_viz:
        print(f"  Output video saved to: {output_dir / (video_path.stem + '_pred.mp4')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    is_yolo = args.model.lower().startswith("yolo")

    if is_yolo:
        run_yolo(
            checkpoint_path=args.checkpoint,
            input_path=input_path,
            output_dir=output_dir,
            score_threshold=args.score_threshold,
            device=args.device,
            imgsz=args.imgsz,
            show=args.show,
        )
        return

    # --- Torchvision path ---
    device = torch.device(args.device)
    model = load_torchvision_model(args.checkpoint, args.model, args.num_classes, device)
    save_viz = not args.no_save_viz

    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        print(f"Found {len(image_paths)} images in {input_path}")
        run_torchvision_images(
            model,
            image_paths,
            device,
            output_dir,
            args.score_threshold,
            args.soft_nms,
            args.nms_threshold,
            save_viz,
            args.show,
        )
    elif suffix in _IMAGE_EXTENSIONS:
        run_torchvision_images(
            model,
            [input_path],
            device,
            output_dir,
            args.score_threshold,
            args.soft_nms,
            args.nms_threshold,
            save_viz,
            args.show,
        )
    elif suffix in _VIDEO_EXTENSIONS:
        run_torchvision_video(
            model,
            input_path,
            device,
            output_dir,
            args.score_threshold,
            args.soft_nms,
            args.nms_threshold,
            save_viz,
            args.show,
        )
    else:
        raise ValueError(f"Unsupported input type: {input_path}")


if __name__ == "__main__":
    main()
