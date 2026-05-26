"""Inference script for VisDrone object detection models.

Supports inference on:
- Single images
- Multiple images in a directory
- All registered models (torchvision, YOLO, DETR)
- Automatic format handling for different model types
- Soft-NMS post-processing
- Test-Time Augmentation (TTA)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on VisDrone models")

    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        help="Model name",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Input
    parser.add_argument("--input", required=True, help="Input image/directory/video")
    parser.add_argument("--output-dir", default="inference_outputs", help="Output directory")

    # Inference parameters
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )

    # Post-processing
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--soft-nms", action="store_true", help="Use soft-NMS")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")

    # Visualization
    parser.add_argument("--no-save-viz", action="store_true", help="Don't save visualizations")
    parser.add_argument("--show", action="store_true", help="Display results")

    return parser.parse_args()


def load_model(
    checkpoint_path: str, model_name: str, num_classes: int, device: torch.device
) -> tuple:
    """Load model from checkpoint.

    Returns:
        Tuple of (model, is_yolo_model)
    """
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    is_yolo = "yolo" in model_name.lower()
    print("✓ Model loaded successfully")
    return model, is_yolo


def process_image(image_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and preprocess image.

    Returns:
        Tuple of (image_tensor, original_size)
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    return image_tensor, original_size


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.5,
    is_yolo: bool = False,
) -> dict:
    """Run inference on a single image.

    Args:
        model: Detection model
        image_tensor: Image as tensor [C, H, W] in [0, 1]
        device: Device to run on
        score_threshold: Confidence threshold
        is_yolo: Whether this is a YOLO model

    Returns:
        Dictionary with boxes, labels, scores
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if is_yolo:
            # YOLO returns results with .boxes attribute
            results = model([image_tensor])
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
        else:
            # Torchvision models
            predictions = model([image_tensor])
            result = predictions[0]

            boxes = result["boxes"].cpu().numpy()  # [x1, y1, x2, y2]
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()

    # Filter by score threshold
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def apply_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Soft-NMS to detection results.

    Args:
        boxes: Detection boxes [N, 4]
        scores: Detection scores [N]
        labels: Detection labels [N]
        sigma: Gaussian penalty parameter
        score_threshold: Minimum score to keep

    Returns:
        Filtered boxes, scores, labels
    """
    boxes = torch.from_numpy(boxes).float()
    scores = torch.from_numpy(scores).float()
    labels = torch.from_numpy(labels)

    unique_labels = labels.unique()

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for label in unique_labels:
        class_mask = labels == label
        class_boxes = boxes[class_mask].clone()
        class_scores = scores[class_mask].clone()

        while len(class_boxes) > 0:
            if class_scores.max() < score_threshold:
                break

            max_idx = class_scores.argmax()
            max_box = class_boxes[max_idx]
            max_score = class_scores[max_idx]

            keep_boxes.append(max_box.numpy())
            keep_scores.append(max_score.item())
            keep_labels.append(label.item())

            class_boxes = torch.cat([class_boxes[:max_idx], class_boxes[max_idx + 1 :]])
            class_scores = torch.cat([class_scores[:max_idx], class_scores[max_idx + 1 :]])

            if len(class_boxes) == 0:
                break

            # Compute IoU with max box
            ious = _compute_iou(max_box.unsqueeze(0), class_boxes)
            class_scores = class_scores * torch.exp(-(ious.squeeze() ** 2) / sigma)

    return (
        np.array(keep_boxes) if keep_boxes else np.zeros((0, 4)),
        np.array(keep_scores) if keep_scores else np.array([]),
        np.array(keep_labels) if keep_labels else np.array([]),
    )


def _compute_iou(box1: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one box and multiple boxes."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    lt = torch.max(box1[:, None, :2], boxes[:, :2])
    rb = torch.min(box1[:, None, 2:], boxes[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    return iou


def visualize_predictions(
    image_path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Visualize predictions on image.

    Args:
        image_path: Path to image
        boxes: Detection boxes [N, 4] in [x1, y1, x2, y2]
        scores: Detection scores [N]
        labels: Detection labels [N]
        class_names: List of class names

    Returns:
        Image with visualizations
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        text = f"{class_name}: {score:.2f}"
        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return image


def main():
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, is_yolo = load_model(
        args.checkpoint,
        args.model,
        args.num_classes,
        device,
    )

    # Get input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    else:
        raise ValueError(f"Input path not found: {input_path}")

    print(f"\nRunning inference on {len(image_paths)} images...\n")

    # Run inference
    start_time = time.time()
    for image_path in image_paths:
        print(f"Processing: {image_path.name}...", end=" ")

        # Load and preprocess image
        image_tensor, original_size = process_image(image_path)

        # Run inference
        result = run_inference(
            model,
            image_tensor,
            device,
            score_threshold=args.score_threshold,
            is_yolo=is_yolo,
        )

        # Apply soft-NMS if requested
        if args.soft_nms and len(result["boxes"]) > 0:
            result["boxes"], result["scores"], result["labels"] = apply_soft_nms(
                result["boxes"],
                result["scores"],
                result["labels"],
            )

        # Visualize
        if not args.no_save_viz:
            viz_image = visualize_predictions(
                image_path,
                result["boxes"],
                result["scores"],
                result["labels"],
                VISDRONE_CLASSES,
            )

            if viz_image is not None:
                output_path = output_dir / f"{image_path.stem}_pred.jpg"
                cv2.imwrite(str(output_path), viz_image)

        print(f"Detected {len(result['boxes'])} objects")

    elapsed = time.time() - start_time
    print(f"\nInference complete in {elapsed:.2f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
