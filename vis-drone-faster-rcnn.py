#!/usr/bin/env python3
"""Validate a TensorFlow SavedModel on images.

Displays or saves annotated results with rich progress bar.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# 🔧 COMPATIBILITY PATCH FOR TF 2.x
if not hasattr(tf, "gfile"):
    tf.gfile = tf.io.gfile
# Also patch python_io if needed
if not hasattr(tf, "python_io"):
    tf.python_io = tf.io

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import fire
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# Import TF OD API utilities
try:
    from object_detection.utils import label_map_util, visualization_utils as viz_utils
except ImportError as e:
    raise ImportError(
        "❌ Could not import 'object_detection'. "
        "Did you run: "
        "`pip install tf-models-official` or install from models/research/?"
    ) from e

console = Console()


class ModelValidator:
    """Validates a TensorFlow SavedModel on one or more images.

    Visualizes detections with bounding boxes.
    """

    def __init__(
        self,
        saved_model_dir: str,
        label_map_path: str,
        min_score_thresh: float = 0.5,
        line_thickness: int = 4,
        category_display_name: bool = True,
    ):
        self.saved_model_dir = Path(saved_model_dir)
        self.label_map_path = Path(label_map_path)
        self.min_score_thresh = min_score_thresh
        self.line_thickness = line_thickness
        self.category_display_name = category_display_name

        self.detect_fn = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self) -> Any:
        """Load the SavedModel."""
        if not (self.saved_model_dir / "saved_model.pb").exists():
            raise FileNotFoundError(f"SavedModel not found at {self.saved_model_dir}")

        console.print(
            f"[bold blue]🧠 Loading model from {self.saved_model_dir}...[/bold blue]"
        )
        model = tf.saved_model.load(str(self.saved_model_dir))
        return model

    def _load_label_map(self) -> Dict[int, Dict[str, Any]]:
        """Load label map and create category index."""
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {self.label_map_path}")

        try:
            label_map = label_map_util.load_labelmap(str(self.label_map_path))
            categories = label_map_util.convert_label_map_to_categories(
                label_map,
                max_num_classes=100,  # Safe upper bound
                use_display_name=self.category_display_name,
            )
            return label_map_util.create_category_index(categories)
        except Exception as e:
            raise RuntimeError(f"Failed to load label map: {e}")

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def predict_and_annotate(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
    ):
        """Run inference and save/display result."""
        image_np = self._load_image(image_path)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, axis=0), dtype=tf.uint8
        )

        # Run inference
        detections = self.detect_fn(input_tensor)

        # Extract and process results
        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(int)
        scores = detections["detection_scores"][0].numpy()

        # Visualize on image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=50,
            min_score_thresh=self.min_score_thresh,
            agnostic_mode=False,
            line_thickness=self.line_thickness,
        )

        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save or show
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image_bgr)
        else:
            cv2.imshow("Detection", image_bgr)
            console.print(
                "[yellow]📌 Press any key in the window to continue...[/yellow]"
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def validate(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        max_images: Optional[int] = None,
        show_preview: bool = False,
    ):
        """Validate model on single image or directory of images.

        Args:
            input_path: Path to image file or directory
            output_dir: Directory to save annotated images. If None, only preview.
            max_images: Limit number of images processed
            show_preview: Show OpenCV window even if saving (can be slow)
        """
        p = Path(input_path)

        if p.is_file():
            image_paths = [p]
        elif p.is_dir():
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_paths.extend(p.glob(ext))
            image_paths = sorted(set(image_paths))
            if max_images:
                image_paths = image_paths[:max_images]
        else:
            raise FileNotFoundError(f"Invalid input path: {input_path}")

        if not image_paths:
            console.print("[yellow]⚠️ No images found.[/yellow]")
            return

        total = len(image_paths)
        success_count = 0

        # Decide output mode
        save_output = output_dir is not None
        out_dir = Path(output_dir) if save_output else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating...", total=total)

            for img_path in image_paths:
                try:
                    if save_output:
                        out_path = out_dir / f"detected_{img_path.name}"
                    else:
                        out_path = None

                    self.predict_and_annotate(img_path, out_path)

                    if show_preview and save_output:
                        # Also show preview
                        temp_img = cv2.imread(str(out_path))
                        cv2.imshow("Saved Detection", temp_img)
                        cv2.waitKey(1)  # Non-blocking wait

                    success_count += 1
                except Exception as e:
                    console.print(f"[red]❌ Failed on {img_path.name}: {e}[/red]")
                finally:
                    progress.advance(task)

        status = "🎉 All good!" if success_count == total else "⚠️ Some failures"
        console.print(
            "[bold green]✅ Validation complete:"
            f" {success_count}/{total} succeeded. {status}[/bold green]"
        )


def main(
    model_dir: str = "inference_graph/saved_model",
    label_map: str = "training/labelmap.pbtxt",
    input_path: str = "test_images/",
    output_dir: Optional[str] = "detection_results/",
    min_score: float = 0.5,
    max_images: Optional[int] = None,
    show_preview: bool = False,
):
    """Validate object detection model on images using SavedModel format.

    Args:
        model_dir: Path to directory containing saved_model.pb
        label_map: Path to labelmap.pbtxt
        input_path: Path to input image or folder of images
        output_dir: Folder to save annotated images.
                    Use --output_dir=null to disable saving.
        min_score: Minimum confidence threshold for visualization
        max_images: Limit number of images to process (useful for testing)
        show_preview: Show live preview window even when saving outputs
    """
    # Handle disabling output
    if output_dir and output_dir.lower() == "null":
        output_dir = None

    validator = ModelValidator(
        saved_model_dir=model_dir,
        label_map_path=label_map,
        min_score_thresh=min_score,
    )
    validator.validate(
        input_path=input_path,
        output_dir=output_dir,
        max_images=max_images,
        show_preview=show_preview,
    )


if __name__ == "__main__":
    fire.Fire(main)
