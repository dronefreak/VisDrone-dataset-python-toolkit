"""Real-time webcam/video demo for VisDrone object detection.

Supports all registered models (torchvision, YOLO) and any OpenCV-compatible
video source: webcam index, video file, or RTSP stream.

Controls:
  'q' — quit          's' — save frame        Space — pause/resume
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model

if TYPE_CHECKING:
    pass  # cv2.Mat is not a real type; we use np.ndarray in signatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time detection demo (webcam / video)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--checkpoint", help="Path to model checkpoint (.pt file)")
    parser.add_argument("--model", default="fasterrcnn_resnet50", help="Model name")
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Source: webcam index OR video/stream URL
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0), video file path, or stream URL",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width (webcam only)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (webcam only)")

    # Inference
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Display
    parser.add_argument("--no-display-fps", action="store_true", help="Hide FPS overlay")
    parser.add_argument("--save-dir", default="webcam_captures", help="Directory for saved frames")

    return parser.parse_args()


class FPSCounter:
    """Sliding-window FPS counter."""

    def __init__(self, window_size: int = 30) -> None:
        self.frame_times: deque[float] = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self) -> None:
        now = time.time()
        self.frame_times.append(now - self.last_time)
        self.last_time = now

    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return float(len(self.frame_times) / sum(self.frame_times))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_torchvision_model(
    checkpoint_path: str | None,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    if checkpoint_path:
        print(f"Loading {model_name} from {checkpoint_path} ...")
        model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print("✓ Checkpoint loaded")
    else:
        print(f"Creating pretrained {model_name} (COCO weights) ...")
        model = get_model(model_name=model_name, num_classes=num_classes, pretrained=True)
        print("✓ Pretrained model loaded")
        print("  Tip: Train on VisDrone for better aerial detection results!")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_torchvision(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    score_threshold: float,
) -> tuple[np.ndarray, int]:
    """Run torchvision model on a BGR frame. Returns (annotated_frame, n_detections)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    preds = model([tensor.to(device)])[0]

    boxes = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    mask = scores >= score_threshold
    return draw_detections(frame_bgr, boxes[mask], labels[mask], scores[mask]), int(mask.sum())


def infer_yolo(
    yolo_model: Any,
    frame_bgr: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, int]:
    """Run YOLO model on a BGR frame. Returns (annotated_frame, n_detections)."""
    results = yolo_model.predict(frame_bgr, conf=score_threshold, verbose=False)
    annotated = results[0].plot()
    return annotated, len(results[0].boxes)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_CLASS_COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 255, 128),
]


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Draw bounding boxes with class-coloured labels."""
    h, w = frame.shape[:2]
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = _CLASS_COLORS[int(label) % len(_CLASS_COLORS)]
        cls_name = VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else f"cls{label}"
        text = f"{cls_name}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly1, ly2 = max(y1 - th - 4, 0), max(y1 - th - 4, 0) + th + 4
        cv2.rectangle(frame, (x1, ly1), (x1 + tw, ly2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, ly2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    is_yolo = args.model.lower().startswith("yolo")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    if is_yolo:
        try:
            from ultralytics import YOLO as UltralyticsYOLO
        except ImportError as err:
            raise ImportError("pip install ultralytics>=8.0.0") from err
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for YOLO models")
        yolo_model = UltralyticsYOLO(args.checkpoint)
        torch_model = None
        print(f"✓ Loaded YOLO model from {args.checkpoint}")
    else:
        torch_model = load_torchvision_model(args.checkpoint, args.model, args.num_classes, device)
        yolo_model = None

    # Open source
    try:
        cam_idx = int(args.source)
        source: int | str = cam_idx
        is_webcam = True
    except ValueError:
        source = args.source
        is_webcam = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source!r}")

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Source opened: {w}×{h}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fps_counter = FPSCounter()

    print("\nControls: 'q' quit | 's' save frame | Space pause/resume\n")

    paused = False
    frame_count = 0
    saved_count = 0
    frame: cv2.Mat | None = None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream.")
                    break
                frame_count += 1

                if is_yolo and yolo_model is not None:
                    annotated, n_det = infer_yolo(yolo_model, frame, args.score_threshold)
                else:
                    assert torch_model is not None
                    annotated, n_det = infer_torchvision(
                        torch_model, frame, device, args.score_threshold
                    )

                fps_counter.update()

                if not args.no_display_fps:
                    cv2.putText(
                        annotated,
                        f"FPS: {fps_counter.get_fps():.1f}  Det: {n_det}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                display_frame = annotated
            else:
                display_frame = frame  # type: ignore[assignment]

            if display_frame is not None:
                cv2.imshow("VisDrone Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and display_frame is not None:
                saved_count += 1
                p = save_dir / f"capture_{saved_count:04d}.jpg"
                cv2.imwrite(str(p), display_frame)
                print(f"✓ Saved {p}")
            elif key == ord(" "):
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(
            f"\nFrames: {frame_count}  Saved: {saved_count}  "
            f"Avg FPS: {fps_counter.get_fps():.1f}"
        )


if __name__ == "__main__":
    main()
