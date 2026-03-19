"""
Evaluation script for VisDrone object detection models.

Computes metrics on validation/test sets.
Supports COCO-style evaluation with pycocotools if available.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.soft_nms_utils import (
    apply_soft_nms_per_class,
    configure_model_for_better_recall,
)

# Import TTA and Soft-NMS utilities
from visdrone_toolkit.tta_utils import tta_inference
from visdrone_toolkit.utils import VISDRONE_CLASSES, collate_fn, compute_metrics, get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VisDrone detection models")

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
            "rfdetr_base",
            "rfdetr_large",
        ],
        help="Model architecture",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Dataset
    parser.add_argument("--image-dir", required=True, help="Images directory")
    parser.add_argument("--annotation-dir", required=True, help="Annotations directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Evaluation options
    parser.add_argument(
        "--score-threshold", type=float, default=0.05, help="Score threshold for detections"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for matching"
    )

    # NEW: TTA and Soft-NMS options
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--soft-nms", action="store_true", help="Use soft-NMS instead of hard NMS")
    parser.add_argument(
        "--lower-threshold", action="store_true", help="Use lower detection threshold (0.01)"
    )

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    # Output
    parser.add_argument("--output-dir", default="eval_outputs", help="Output directory")
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions to JSON file"
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
    lower_threshold: bool = False,
):
    """Load model from checkpoint with proper architecture modifications."""
    print(f"Loading model from {checkpoint_path}...")

    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    # Apply small anchor modifications for Faster R-CNN
    if model_name in ["fasterrcnn_resnet50", "fasterrcnn_mobilenet"]:
        print("Applying small anchor modifications...")
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        if hasattr(model, "rpn") and hasattr(model.rpn, "anchor_generator"):
            # Small anchors: 16, 32, 64, 128, 256
            small_anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(small_anchor_sizes)
            model.rpn.anchor_generator = AnchorGenerator(
                sizes=small_anchor_sizes, aspect_ratios=aspect_ratios
            )

            # Update RPN parameters
            model.rpn.pre_nms_top_n_train = 2000
            model.rpn.post_nms_top_n_train = 2000
            model.rpn.pre_nms_top_n_test = 1000
            model.rpn.post_nms_top_n_test = 1000

            # NMS settings
            model.roi_heads.nms_thresh = 0.3
            model.roi_heads.score_thresh = 0.05
            model.roi_heads.detections_per_img = 300

            print("✓ Small anchors and NMS settings applied")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    # Apply lower threshold configuration if requested
    if lower_threshold:
        model = configure_model_for_better_recall(model, model_name)

    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    use_tta: bool = False,
    use_soft_nms: bool = False,
) -> Dict:
    """Evaluate model on dataset with optional TTA and Soft-NMS."""
    print(f"\n{'=' * 60}")
    print("Running Evaluation")
    if use_tta:
        print("  Using Test-Time Augmentation (TTA)")
    if use_soft_nms:
        print("  Using Soft-NMS")
    print(f"{'=' * 60}")

    all_predictions = []
    all_targets = []
    total_inference_time = 0.0
    num_images = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        batch_start = time.time()

        for img, target in zip(images, targets):
            # Use TTA if enabled
            if use_tta:
                pred = tta_inference(model, img, device, score_threshold)
            else:
                # Standard inference
                pred = model([img.to(device)])[0]

                # Filter by score threshold
                mask = pred["scores"] >= score_threshold
                pred = {
                    "boxes": pred["boxes"][mask],
                    "labels": pred["labels"][mask],
                    "scores": pred["scores"][mask],
                }

            # Apply soft-NMS if enabled
            if use_soft_nms and len(pred["boxes"]) > 0:
                boxes, labels, scores = apply_soft_nms_per_class(
                    pred["boxes"].cpu(),
                    pred["labels"].cpu(),
                    pred["scores"].cpu(),
                    iou_threshold=0.5,
                    sigma=0.5,
                    score_threshold=score_threshold,
                )
                pred = {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }

            all_predictions.append(pred)
            all_targets.append(target)
            num_images += 1

        inference_time = time.time() - batch_start
        total_inference_time += inference_time

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {num_images} images...")

    print(f"\nTotal images evaluated: {num_images}")
    print(f"Average inference time: {(total_inference_time / num_images) * 1000:.2f}ms")
    print(f"Average FPS: {num_images / total_inference_time:.2f}")

    # Compute metrics
    print(f"\n{'=' * 60}")
    print("Computing Metrics")
    print(f"{'=' * 60}")

    metrics = compute_metrics(all_predictions, all_targets, iou_threshold)

    # Print overall metrics
    print(f"\nOverall Metrics (IoU={iou_threshold}):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")

    # Compute per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'=' * 60}")

    per_class_metrics = compute_per_class_metrics(all_predictions, all_targets, iou_threshold)

    for class_idx, class_metrics in sorted(per_class_metrics.items()):
        class_name = (
            VISDRONE_CLASSES[class_idx]
            if class_idx < len(VISDRONE_CLASSES)
            else f"class_{class_idx}"
        )
        print(f"\n{class_name} (class {class_idx}):")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1-Score: {class_metrics['f1']:.4f}")
        print(f"  Ground truth instances: {class_metrics['gt_count']}")
        print(f"  Predicted instances: {class_metrics['pred_count']}")

    return {
        "overall_metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "predictions": all_predictions,
        "targets": all_targets,
        "inference_time": total_inference_time,
        "num_images": num_images,
    }


def compute_per_class_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[int, Dict]:
    """Compute per-class metrics."""
    from visdrone_toolkit.utils import box_iou

    # Collect all class indices
    all_classes = set()
    for target in targets:
        all_classes.update(target["labels"].cpu().numpy().tolist())

    per_class_metrics = {}

    for class_idx in sorted(all_classes):
        tp = 0
        fp = 0
        fn = 0
        gt_count = 0
        pred_count = 0

        for pred, target in zip(predictions, targets):
            # Filter by class
            pred_mask = pred["labels"].cpu() == class_idx
            target_mask = target["labels"].cpu() == class_idx

            pred_boxes = pred["boxes"].cpu()[pred_mask]
            target_boxes = target["boxes"].cpu()[target_mask]

            gt_count += len(target_boxes)
            pred_count += len(pred_boxes)

            if len(pred_boxes) == 0 and len(target_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                fn += len(target_boxes)
                continue
            elif len(target_boxes) == 0:
                fp += len(pred_boxes)
                continue

            # Compute IoU
            ious = box_iou(pred_boxes, target_boxes)

            # Match predictions to targets
            matched_targets = set()
            for i in range(len(pred_boxes)):
                max_iou, max_idx = ious[i].max(dim=0)
                if max_iou >= iou_threshold:
                    if max_idx.item() not in matched_targets:
                        tp += 1
                        matched_targets.add(max_idx.item())
                    else:
                        fp += 1
                else:
                    fp += 1

            fn += len(target_boxes) - len(matched_targets)

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_idx] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": gt_count,
            "pred_count": pred_count,
        }

    return per_class_metrics


def save_results(results: Dict, output_dir: Path, save_predictions: bool):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_data = {
        "overall_metrics": results["overall_metrics"],
        "per_class_metrics": {
            int(k): {
                key: float(val) if isinstance(val, (np.floating, float)) else int(val)
                for key, val in v.items()
            }
            for k, v in results["per_class_metrics"].items()
        },
        "inference_time": results["inference_time"],
        "num_images": results["num_images"],
        "avg_inference_time_ms": (results["inference_time"] / results["num_images"]) * 1000,
        "fps": results["num_images"] / results["inference_time"],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n✓ Metrics saved to {metrics_path}")

    # Save predictions if requested
    if save_predictions:
        predictions_path = output_dir / "predictions.json"
        predictions_data = []

        for _, (pred, target) in enumerate(zip(results["predictions"], results["targets"])):
            predictions_data.append(
                {
                    "image_id": int(target["image_id"][0]),
                    "predictions": {
                        "boxes": pred["boxes"].cpu().numpy().tolist(),
                        "labels": pred["labels"].cpu().numpy().tolist(),
                        "scores": pred["scores"].cpu().numpy().tolist(),
                    },
                    "ground_truth": {
                        "boxes": target["boxes"].cpu().numpy().tolist(),
                        "labels": target["labels"].cpu().numpy().tolist(),
                    },
                }
            )

        with open(predictions_path, "w") as f:
            json.dump(predictions_data, f, indent=2)

        print(f"✓ Predictions saved to {predictions_path}")


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = VisDroneDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        filter_ignored=True,
        filter_crowd=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Load model
    model = load_model(
        args.checkpoint, args.model, args.num_classes, device, lower_threshold=args.lower_threshold
    )

    # Evaluate
    results = evaluate_model(
        model,
        data_loader,
        device,
        args.score_threshold,
        args.iou_threshold,
        use_tta=args.tta,
        use_soft_nms=args.soft_nms,
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, args.save_predictions)

    print(f"\n{'=' * 60}")
    print("Evaluation completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
