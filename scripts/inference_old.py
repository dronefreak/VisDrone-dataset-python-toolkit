"""
Inference script for VisDrone object detection models.

Supports inference on:
- Single images
- Multiple images in a directory
- Video files
- Test-Time Augmentation (TTA)
- Soft-NMS post-processing
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from visdrone_toolkit.utils import VISDRONE_CLASSES, get_model
from visdrone_toolkit.visualization import visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on VisDrone models")

    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        choices=[
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50",
        ],
        help="Model architecture",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Input
    parser.add_argument("--input", required=True, help="Input image/directory/video")
    parser.add_argument("--output-dir", default="inference_outputs", help="Output directory")

    # Inference parameters
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    # Post-processing options
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--soft-nms", action="store_true", help="Use soft-NMS instead of hard NMS")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")

    # Visualization
    parser.add_argument("--no-save-viz", action="store_true", help="Don't save visualizations")
    parser.add_argument("--show", action="store_true", help="Display results")

    return parser.parse_args()


def load_model(checkpoint_path: str, model_name: str, num_classes: int, device: torch.device):
    """Load model from checkpoint."""
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
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


def apply_soft_nms(boxes, scores, labels, sigma=0.5, score_threshold=0.001):
    """
    Apply Soft-NMS to detection results.

    Args:
        boxes: Detection boxes
        scores: Detection scores
        labels: Detection labels
        nms_threshold: IoU threshold (for compatibility, not used in pure Soft-NMS)
        sigma: Gaussian penalty parameter (lower = more aggressive suppression)
        score_threshold: Minimum score to keep after penalty

    Returns filtered boxes, scores, and labels.
    """
    # Convert to tensors if needed
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Get unique classes
    unique_labels = labels.unique()

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for label in unique_labels:
        # Filter by class
        class_mask = labels == label
        class_boxes = boxes[class_mask].clone()
        class_scores = scores[class_mask].clone()

        # Apply Soft-NMS per class
        while len(class_boxes) > 0:
            if class_scores.max() < score_threshold:
                break

            max_idx = class_scores.argmax()
            max_box = class_boxes[max_idx]
            max_score = class_scores[max_idx]

            # Keep the max scoring box
            keep_boxes.append(max_box)
            keep_scores.append(max_score)
            keep_labels.append(label)

            # Remove max box
            class_boxes = torch.cat([class_boxes[:max_idx], class_boxes[max_idx + 1 :]])
            class_scores = torch.cat([class_scores[:max_idx], class_scores[max_idx + 1 :]])

            if len(class_boxes) == 0:
                break

            # Compute IoU with remaining boxes
            ious = torchvision.ops.box_iou(max_box.unsqueeze(0), class_boxes)[0]

            # Apply Gaussian penalty (pure Soft-NMS)
            weights = torch.exp(-(ious**2) / sigma)
            class_scores = class_scores * weights

    if len(keep_boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.stack(keep_boxes), torch.stack(keep_scores), torch.stack(keep_labels)


@torch.no_grad()
def run_inference_with_tta(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.5,
) -> dict:
    """
    Run inference with test-time augmentation.

    Averages predictions across:
    - Original image
    - Horizontal flip
    - Multi-scale (0.8x, 1.0x, 1.2x)
    """
    h, w = image_tensor.shape[1:]
    all_boxes = []
    all_scores = []
    all_labels = []

    # Scales for multi-scale TTA
    scales = [0.8, 1.0, 1.2]

    for scale in scales:
        # Resize image
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            )[0]
        else:
            scaled_img = image_tensor

        # Original + horizontal flip
        for flip in [False, True]:
            test_img = torch.flip(scaled_img, dims=[2]) if flip else scaled_img

            # Run inference
            predictions = model([test_img.to(device)])[0]

            boxes = predictions["boxes"].cpu()
            scores = predictions["scores"].cpu()
            labels = predictions["labels"].cpu()

            # Unflip boxes if needed
            if flip:
                img_w = test_img.shape[2]
                boxes[:, [0, 2]] = img_w - boxes[:, [2, 0]]

            # Unscale boxes if needed
            if scale != 1.0:
                boxes = boxes / scale

            # Filter by score
            mask = scores >= score_threshold
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_labels.append(labels[mask])

    # Concatenate all predictions
    if len(all_boxes) > 0 and sum(len(b) for b in all_boxes) > 0:
        final_boxes = torch.cat([b for b in all_boxes if len(b) > 0])
        final_scores = torch.cat([s for s in all_scores if len(s) > 0])
        final_labels = torch.cat([l for l in all_labels if len(l) > 0])  # noqa: E741
    else:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty(0)
        final_labels = torch.empty(0, dtype=torch.long)

    return {
        "boxes": final_boxes,
        "labels": final_labels,
        "scores": final_scores,
    }


@torch.no_grad()
def run_inference_on_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    score_threshold: float = 0.5,
    use_tta: bool = False,
    use_soft_nms: bool = False,
) -> dict:
    """Run inference on a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to tensor
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    # Run inference
    start_time = time.time()

    if use_tta:
        predictions = run_inference_with_tta(model, image_tensor, device, score_threshold)
    else:
        predictions = model([image_tensor.to(device)])[0]
        predictions = {
            "boxes": predictions["boxes"].cpu(),
            "labels": predictions["labels"].cpu(),
            "scores": predictions["scores"].cpu(),
        }

    inference_time = time.time() - start_time

    # Apply Soft-NMS if enabled
    if use_soft_nms:
        boxes, scores, labels = apply_soft_nms(
            predictions["boxes"],
            predictions["scores"],
            predictions["labels"],
            sigma=0.5,
        )
        predictions = {"boxes": boxes, "labels": labels, "scores": scores}

    # Filter by score threshold
    mask = predictions["scores"] >= score_threshold
    predictions = {
        "boxes": predictions["boxes"][mask],
        "labels": predictions["labels"][mask],
        "scores": predictions["scores"][mask],
    }

    return {
        "predictions": predictions,
        "image": image_np,
        "inference_time": inference_time,
    }


def process_images(
    model: torch.nn.Module,
    input_path: str | Path,
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
    save_viz: bool,
    show: bool,
    use_tta: bool = False,
    use_soft_nms: bool = False,
    nms_threshold: float = 0.5,
):
    """Process images from file or directory."""
    input_path = Path(input_path)

    # Get image files
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            [
                f
                for f in input_path.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if len(image_files) == 0:
        print("No images found!")
        return

    print(f"\nProcessing {len(image_files)} images...")
    print(f"{'=' * 60}")

    total_inference_time = 0
    total_detections = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")

        # Run inference
        result = run_inference_on_image(
            model,
            image_path,
            device,
            score_threshold,
            use_tta=use_tta,
            use_soft_nms=use_soft_nms,
            nms_threshold=nms_threshold,
        )

        num_detections = len(result["predictions"]["boxes"])
        total_detections += num_detections
        total_inference_time += result["inference_time"]

        print(f"  Detections: {num_detections}")
        print(f"  Inference time: {result['inference_time'] * 1000:.2f}ms")

        # Visualize and save
        if save_viz:
            output_path = output_dir / f"{image_path.stem}_result.jpg"
            visualize_predictions(
                result["image"],
                result["predictions"]["boxes"],
                result["predictions"]["labels"],
                result["predictions"]["scores"],
                score_threshold=score_threshold,
                save_path=output_path,
                show=show,
            )
            print(f"  ✓ Saved to {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average inference time: {(total_inference_time / len(image_files)) * 1000:.2f}ms")
    print(f"  FPS: {len(image_files) / total_inference_time:.2f}")


def process_video(
    model: torch.nn.Module,
    video_path: str | Path,
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
):
    """Process video file."""
    video_path = Path(video_path)
    output_path = Path(output_dir) / f"{video_path.stem}_result.mp4"

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    total_inference_time = 0.0

    print(f"\n{'=' * 60}")
    print("Processing frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(device)

            # Run inference
            start_time = time.time()
            predictions = model([image_tensor])[0]
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Filter by score
            mask = predictions["scores"] >= score_threshold
            boxes = predictions["boxes"][mask].cpu().numpy()
            labels = predictions["labels"][mask].cpu().numpy()
            scores = predictions["scores"][mask].cpu().numpy()

            # Draw detections
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)

                # Get class name and color
                class_name = (
                    VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else f"class_{label}"
                )
                color = (0, 255, 0)  # Green

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label_text = f"{class_name}: {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # Write frame
            out.write(frame)

            # Print progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                print(
                    f"  Frame {frame_count}/{total_frames} - "
                    f"Avg FPS: {avg_fps:.2f} - "
                    f"Detections: {len(boxes)}"
                )

    finally:
        cap.release()
        out.release()

    print(f"\n{'=' * 60}")
    print(f"✓ Video saved to {output_path}")
    print(f"  Processed {frame_count} frames")
    print(f"  Average inference FPS: {frame_count / total_inference_time:.2f}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.model, args.num_classes, device)

    # Print inference options
    if args.tta:
        print("✓ Using Test-Time Augmentation (6 augmentations: 3 scales × 2 flips)")
    if args.soft_nms:
        print(f"✓ Using Soft-NMS (threshold={args.nms_threshold})")

    # Check input type
    input_path = Path(args.input)

    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    # Process based on input type
    if input_path.is_file():
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            # Video file
            process_video(model, input_path, output_dir, device, args.score_threshold)
        else:
            # Single image
            process_images(
                model,
                input_path,
                output_dir,
                device,
                args.score_threshold,
                not args.no_save_viz,
                args.show,
                use_tta=args.tta,
                use_soft_nms=args.soft_nms,
                nms_threshold=args.nms_threshold,
            )
    elif input_path.is_dir():
        # Directory of images
        process_images(
            model,
            input_path,
            output_dir,
            device,
            args.score_threshold,
            not args.no_save_viz,
            args.show,
            use_tta=args.tta,
            use_soft_nms=args.soft_nms,
            nms_threshold=args.nms_threshold,
        )
    else:
        raise ValueError(f"Invalid input: {input_path}")

    print(f"\n{'=' * 60}")
    print("Inference completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
