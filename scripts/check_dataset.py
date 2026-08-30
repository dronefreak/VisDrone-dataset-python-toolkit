#!/usr/bin/env python3
"""
Dataset Sanity Check CLI
Scans VisDrone annotation directory and reports integrity issues.

Usage:
    python scripts/check_dataset.py --annotations_dir /path/to/annotations --images_dir /path/to/images
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple


def load_annotation(file_path: str) -> List[List[float]]:
    """
    Load VisDrone annotation file.
    Format: x,y,width,height,confidence,class_id,truncation,occlusion
    """
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    x, y, w, h = map(float, parts[:4])
                    class_id = int(parts[5])
                    boxes.append([x, y, w, h, class_id])
                except ValueError:
                    continue
    return boxes


def check_empty_files(annotations_dir: str) -> List[str]:
    """
    Check for empty annotation files.
    Returns list of empty file paths.
    """
    empty_files = []
    for file_path in Path(annotations_dir).glob("*.txt"):
        if file_path.stat().st_size == 0:
            empty_files.append(str(file_path))
    return empty_files


def check_out_of_bounds_boxes(annotations_dir: str, images_dir: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """
    Check if any boxes are out of image bounds.
    Returns list of (image_file, [(box_index, error_message)])
    """
    issues = []
    
    # Get image dimensions (assuming images are in images_dir)
    # For simplicity, we'll check if boxes are within [0, 1] range
    # since VisDrone uses normalized coordinates (relative to image size)
    
    for ann_file in Path(annotations_dir).glob("*.txt"):
        boxes = load_annotation(str(ann_file))
        file_issues = []
        
        for idx, (x, y, w, h, class_id) in enumerate(boxes):
            # Check if box is within bounds (normalized coordinates should be 0-1)
            if x < 0 or x > 1 or y < 0 or y > 1:
                file_issues.append((idx, f"Box {idx}: x={x}, y={y} out of bounds"))
            if w <= 0 or h <= 0:
                file_issues.append((idx, f"Box {idx}: width={w}, height={h} invalid"))
            if x + w > 1 or y + h > 1:
                file_issues.append((idx, f"Box {idx}: x+w={x+w}, y+h={y+h} out of bounds"))
        
        if file_issues:
            issues.append((str(ann_file), file_issues))
    
    return issues


def check_class_ids(annotations_dir: str, valid_classes: set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) -> List[Tuple[str, List[int]]]:
    """
    Check if class IDs are valid (0-9 for VisDrone).
    Returns list of (file_path, [invalid_class_ids])
    """
    issues = []
    
    for ann_file in Path(annotations_dir).glob("*.txt"):
        boxes = load_annotation(str(ann_file))
        invalid_classes = []
        
        for _, _, _, _, class_id in boxes:
            if class_id not in valid_classes:
                invalid_classes.append(class_id)
        
        if invalid_classes:
            issues.append((str(ann_file), list(set(invalid_classes))))
    
    return issues


def check_missing_annotations(images_dir: str, annotations_dir: str) -> List[str]:
    """
    Check for images that have no corresponding annotation file.
    Returns list of image paths with missing annotations.
    """
    missing = []
    
    # Get all image files (common formats)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
    
    # Check if annotation exists for each image
    for img_path in image_files:
        ann_path = Path(annotations_dir) / f"{img_path.stem}.txt"
        if not ann_path.exists():
            missing.append(str(img_path))
    
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Check VisDrone dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_dataset.py --annotations_dir data/annotations --images_dir data/images
  python scripts/check_dataset.py -a data/annotations -i data/images --verbose
        """
    )
    parser.add_argument(
        "-a", "--annotations_dir",
        required=True,
        help="Path to directory containing annotation files (.txt)"
    )
    parser.add_argument(
        "-i", "--images_dir",
        required=True,
        help="Path to directory containing image files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.annotations_dir):
        print(f"❌ Error: Annotations directory '{args.annotations_dir}' does not exist")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"❌ Error: Images directory '{args.images_dir}' does not exist")
        return 1
    
    print("🔍 Scanning VisDrone dataset...\n")
    
    # Check 1: Empty annotation files
    print("📄 Checking for empty annotation files...")
    empty_files = check_empty_files(args.annotations_dir)
    if empty_files:
        print(f"   ⚠️ Found {len(empty_files)} empty annotation file(s):")
        for f in empty_files[:10]:  # Show first 10
            print(f"      - {f}")
        if len(empty_files) > 10:
            print(f"      ... and {len(empty_files) - 10} more")
    else:
        print("   ✅ No empty annotation files found")
    
    # Check 2: Out-of-bounds boxes
    print("\n📐 Checking for out-of-bounds boxes...")
    oob_issues = check_out_of_bounds_boxes(args.annotations_dir, args.images_dir)
    if oob_issues:
        print(f"   ⚠️ Found {len(oob_issues)} file(s) with out-of-bounds boxes:")
        for file_path, issues in oob_issues[:5]:
            print(f"      - {os.path.basename(file_path)}: {len(issues)} issue(s)")
            if args.verbose:
                for idx, msg in issues[:3]:
                    print(f"          • {msg}")
        if len(oob_issues) > 5:
            print(f"      ... and {len(oob_issues) - 5} more")
    else:
        print("   ✅ No out-of-bounds boxes found")
    
    # Check 3: Invalid class IDs
    print("\n🏷️ Checking for invalid class IDs...")
    class_issues = check_class_ids(args.annotations_dir)
    if class_issues:
        print(f"   ⚠️ Found {len(class_issues)} file(s) with invalid class IDs:")
        for file_path, invalid_ids in class_issues[:5]:
            print(f"      - {os.path.basename(file_path)}: invalid class IDs {invalid_ids}")
        if len(class_issues) > 5:
            print(f"      ... and {len(class_issues) - 5} more")
    else:
        print("   ✅ All class IDs are valid (0-9)")
    
    # Check 4: Missing annotations
    print("\n🖼️ Checking for images without annotations...")
    missing_ann = check_missing_annotations(args.images_dir, args.annotations_dir)
    if missing_ann:
        print(f"   ⚠️ Found {len(missing_ann)} image(s) with no annotation file:")
        for f in missing_ann[:10]:
            print(f"      - {os.path.basename(f)}")
        if len(missing_ann) > 10:
            print(f"      ... and {len(missing_ann) - 10} more")
    else:
        print("   ✅ All images have corresponding annotation files")
    
    # Summary
    print("\n" + "=" * 50)
    total_issues = len(empty_files) + len(oob_issues) + len(class_issues) + len(missing_ann)
    if total_issues == 0:
        print("🎉 No issues found! Dataset is clean.")
    else:
        print(f"📊 Summary: Found {total_issues} total issue(s)")
        print(f"   - Empty annotation files: {len(empty_files)}")
        print(f"   - Out-of-bounds boxes: {len(oob_issues)}")
        print(f"   - Invalid class IDs: {len(class_issues)}")
        print(f"   - Missing annotations: {len(missing_ann)}")
    
    return 0


if __name__ == "__main__":
    exit(main())