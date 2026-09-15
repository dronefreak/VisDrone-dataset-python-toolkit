"""
COCO to VisDrone annotation converter.

Converts COCO JSON format annotations to VisDrone format.

COCO format:
    {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"image_id": 1, "bbox": [x, y, width, height], "category_id": 1}
        ],
        "categories": [{"id": 1, "name": "pedestrian"}]
    }

VisDrone format (per line):
    x, y, width, height, confidence, class_id, truncation, occlusion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Mapping from COCO category names to VisDrone class IDs (0-9)
COCO_TO_VISDRONE_CLASSES: dict[str, int] = {
    "pedestrian": 0,
    "person": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
    "motorcycle": 9,
}


def coco_to_visdrone(
    coco_json_path: str | Path,
    output_dir: str | Path,
    copy_images: bool = False,
    image_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Convert COCO JSON annotations to VisDrone format.

    Args:
        coco_json_path: Path to COCO JSON file.
        output_dir: Directory to write VisDrone annotation files.
        copy_images: Whether to copy images to output_dir.
        image_dir: Source directory for images (required if copy_images=True).

    Returns:
        Dictionary with conversion statistics:
        - num_images: Number of images processed
        - num_annotations: Total number of annotations
        - num_skipped: Number of annotations skipped (unknown class)
    """
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Build category ID -> name mapping
    category_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
    }

    # Build image ID -> image info mapping
    image_id_to_info: dict[int, dict[str, Any]] = {
        img["id"]: img for img in coco_data.get("images", [])
    }

    # Group annotations by image
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    stats: dict[str, Any] = {
        "num_images": 0,
        "num_annotations": 0,
        "num_skipped": 0,
    }

    # Process each image
    for image_id, image_info in image_id_to_info.items():
        file_name = image_info["file_name"]
        stem = Path(file_name).stem
        output_file = output_dir / f"{stem}.txt"

        annotations = annotations_by_image.get(image_id, [])

        lines: list[str] = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, width, height]
            category_id = ann["category_id"]
            category_name = category_id_to_name.get(category_id, "")

            # Map COCO class to VisDrone class ID
            visdrone_class_id = COCO_TO_VISDRONE_CLASSES.get(category_name.lower())

            if visdrone_class_id is None:
                stats["num_skipped"] += 1
                continue

            x, y, w, h = bbox
            # VisDrone format: x, y, w, h, confidence, class_id, truncation, occlusion
            line = f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{visdrone_class_id},0,0"
            lines.append(line)

        # Write annotation file
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        stats["num_images"] += 1
        stats["num_annotations"] += len(lines)

        # Optionally copy images
        if copy_images and image_dir is not None:
            import shutil

            src = Path(image_dir) / file_name
            if src.exists():
                shutil.copy(src, output_dir / file_name)

    return stats


def validate_visdrone_format(annotation_dir: str | Path) -> bool:
    """
    Validate VisDrone format annotation files.

    Args:
        annotation_dir: Path to VisDrone annotations directory.

    Returns:
        True if valid, False otherwise.
    """
    annotation_dir = Path(annotation_dir)
    ann_files = list(annotation_dir.glob("*.txt"))

    if len(ann_files) == 0:
        print(f"No annotation files found in {annotation_dir}")
        return False

    print(f"Validating {len(ann_files)} VisDrone annotation files...")

    valid_count = 0
    error_count = 0

    for ann_file in ann_files:
        try:
            with open(ann_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) != 8:
                        print(
                            f"Error in {ann_file.name} line {line_num}: "
                            f"Expected 8 values, got {len(parts)}"
                        )
                        error_count += 1
                        continue

                    x, y, w, h = map(float, parts[:4])
                    class_id = int(parts[5])

                    if w <= 0 or h <= 0:
                        print(f"Error in {ann_file.name} line {line_num}: Invalid bbox")
                        error_count += 1
                        continue

                    if not (0 <= class_id <= 9):
                        print(f"Error in {ann_file.name} line {line_num}: Invalid class ID")
                        error_count += 1
                        continue

            valid_count += 1

        except Exception as e:
            print(f"Error validating {ann_file.name}: {e}")
            error_count += 1

    print("\nValidation complete:")
    print(f"Valid files: {valid_count}")
    print(f"Errors: {error_count}")

    return error_count == 0


def main() -> int:
    """Main entry point for COCO to VisDrone conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to VisDrone format",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to COCO JSON file")
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for VisDrone annotations"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output directory",
    )
    parser.add_argument(
        "--image-dir",
        help="Source directory for images (required if --copy-images)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate output")

    args = parser.parse_args()

    stats = coco_to_visdrone(
        coco_json_path=args.input,
        output_dir=args.output,
        copy_images=args.copy_images,
        image_dir=args.image_dir,
    )

    print("✅ Conversion complete!")
    print(f"   Images processed: {stats['num_images']}")
    print(f"   Annotations: {stats['num_annotations']}")
    print(f"   Skipped: {stats['num_skipped']}")

    if args.validate:
        validate_visdrone_format(args.output)

    return 0


if __name__ == "__main__":
    exit(main())
