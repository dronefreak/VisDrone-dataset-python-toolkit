"""
Soft-NMS and improved detection utilities.

Soft-NMS reduces scores of overlapping boxes instead of removing them,
which is better for dense scenes.

"""

import torch


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS: Instead of removing overlapping boxes, reduce their scores.

    Better for dense scenes where objects overlap.

    Args:
        boxes: [N, 4] tensor of boxes
        scores: [N] tensor of scores
        iou_threshold: IoU threshold for suppression
        sigma: Gaussian penalty factor (higher = softer)
        score_threshold: Minimum score to keep

    Returns:
        keep: Indices of boxes to keep
        scores: Updated scores (some reduced by soft-NMS)
    """
    # Convert to numpy for easier manipulation (device-agnostic)
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy().copy()

    N = len(boxes)
    keep = []

    # Sort by score descending
    order = scores.argsort()[::-1]

    for i in range(N):
        idx = order[i]

        if scores[idx] < score_threshold:
            continue

        keep.append(idx)

        # Compute IoU with remaining boxes
        for j in range(i + 1, N):
            jdx = order[j]

            if scores[jdx] < score_threshold:
                continue

            iou = compute_iou(boxes[idx], boxes[jdx])

            # Soft-NMS: reduce score based on IoU
            # Gaussian penalty when IoU exceeds threshold
            if iou > iou_threshold:
                weight = float(torch.exp(-torch.tensor(iou**2) / sigma))
                scores[jdx] *= weight

    return torch.tensor(keep), torch.from_numpy(scores)


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def apply_soft_nms_per_class(
    boxes, labels, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001
):
    """
    Apply soft-NMS per class.

    Args:
        boxes: [N, 4] tensor
        labels: [N] tensor
        scores: [N] tensor
        iou_threshold: IoU threshold
        sigma: Soft-NMS sigma
        score_threshold: Minimum score

    Returns:
        Filtered boxes, labels, scores
    """
    keep_indices: list[torch.Tensor] = []
    updated_scores = scores.clone()

    for class_id in labels.unique():
        class_mask = labels == class_id
        class_indices = torch.where(class_mask)[0]
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Apply soft-NMS
        keep, new_scores = soft_nms(
            class_boxes,
            class_scores,
            iou_threshold=iou_threshold,
            sigma=sigma,
            score_threshold=score_threshold,
        )

        # Update scores
        if len(keep) > 0:
            original_indices = class_indices[keep.cpu()]
            keep_indices.append(original_indices)
            updated_scores[original_indices] = new_scores[keep]

    if len(keep_indices) > 0:
        keep_indices = torch.cat(keep_indices)  # type: ignore[assignment]
        return boxes[keep_indices], labels[keep_indices], updated_scores[keep_indices]
    else:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), torch.zeros((0,))


def configure_model_for_better_recall(model, model_name="fasterrcnn_resnet50"):
    """
    Configure model for better recall (lower thresholds, more detections).

    Apply this after loading the model checkpoint.

    Args:
        model: PyTorch detection model
        model_name: Model architecture name

    Returns:
        Modified model
    """
    if model_name in ["fasterrcnn_resnet50", "fasterrcnn_mobilenet"]:
        # Lower score threshold for ROI heads
        if hasattr(model, "roi_heads"):
            model.roi_heads.score_thresh = 0.01  # From 0.05
            model.roi_heads.nms_thresh = 0.3  # Keep same
            model.roi_heads.detections_per_img = 500  # From 300

            print("✓ Configured for better recall:")
            print("  - Score threshold: 0.01 (was 0.05)")
            print("  - Detections per image: 500 (was 300)")

    elif model_name == "fcos_resnet50":
        # FCOS configuration
        if hasattr(model, "head"):
            model.head.score_thresh = 0.01
            model.head.nms_thresh = 0.3
            model.head.detections_per_img = 500

            print("✓ Configured FCOS for better recall")

    elif model_name == "retinanet_resnet50":
        # RetinaNet configuration
        if hasattr(model, "head"):
            model.head.score_thresh = 0.01
            model.head.nms_thresh = 0.3
            model.head.detections_per_img = 500

            print("✓ Configured RetinaNet for better recall")
    else:
        print(f"⚠️  Model '{model_name}' not recognized for recall configuration.")

    return model
