#!/usr/bin/env python3
"""Convert VisDrone .txt annotations to Pascal VOC XML format.

Supports multiple image formats, rich progress bar, and CLI args via Fire.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import fire
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()


class VisDroneToVOCConverter:
    """Converts VisDrone annotation format (.txt) to Pascal VOC XML format.

    Supports multiple image types and optional bounding box drawing.
    """

    # Default label map from VisDrone
    DEFAULT_LABEL_DICT = {
        "0": "Ignore",
        "1": "Pedestrian",
        "2": "People",
        "3": "Bicycle",
        "4": "Car",
        "5": "Van",
        "6": "Truck",
        "7": "Tricycle",
        "8": "Awning-tricycle",
        "9": "Bus",
        "10": "Motor",
        "11": "Others",
    }

    # Supported image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        input_img_folder: str,
        input_ann_folder: str,
        output_img_folder: str,
        output_ann_folder: str,
        label_dict: Optional[Dict[str, str]] = None,
        draw_bboxes: bool = False,
        color: tuple = (255, 0, 0),
        thickness: int = 2,
    ):
        self.input_img_folder = Path(input_img_folder)
        self.input_ann_folder = Path(input_ann_folder)
        self.output_img_folder = Path(output_img_folder)
        self.output_ann_folder = Path(output_ann_folder)

        self.label_dict = label_dict or self.DEFAULT_LABEL_DICT
        self.draw_bboxes = draw_bboxes
        self.color = color
        self.thickness = thickness

        self._validate_paths()

    def _validate_paths(self):
        """Ensure required directories exist."""
        if not self.input_img_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.input_img_folder}")
        if not self.input_ann_folder.is_dir():
            raise FileNotFoundError(
                f"Annotation folder not found: {self.input_ann_folder}"
            )

        self.output_img_folder.mkdir(parents=True, exist_ok=True)
        self.output_ann_folder.mkdir(parents=True, exist_ok=True)

    def _find_image_file(self, stem: str) -> Optional[Path]:
        """Search for image with any supported extension."""
        for ext in self.IMAGE_EXTENSIONS:
            img_path = self.input_img_folder / f"{stem}{ext}"
            if img_path.exists():
                return img_path
        return None

    def create_voc_xml(
        self,
        filename: str,
        img_path: str,
        width: int,
        height: int,
        depth: int,
        objects: List[Dict],
    ) -> str:
        """Generate Pascal VOC-formatted XML string."""
        header = f"""<annotation>
	<folder>{self.output_img_folder.name}</folder>
	<filename>{filename}</filename>
	<path>{img_path}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>{depth}</depth>
	</size>
	<segmented>0</segmented>"""

        obj_strings = []
        for obj in objects:
            obj_str = f"""
	<object>
		<name>{obj["name"]}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{obj["xmin"]}</xmin>
			<ymin>{obj["ymin"]}</ymin>
			<xmax>{obj["xmax"]}</xmax>
			<ymax>{obj["ymax"]}</ymax>
		</bndbox>
	</object>"""
            obj_strings.append(obj_str)

        footer = "\n</annotation>"
        return header + "".join(obj_strings) + footer

    def process_annotation_file(
        self,
        txt_path: Path,
        img_path: Path,
        xml_output_path: Path,
        img_output_path: Path,
    ):
        """Process a single annotation-image pair."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        h, w, d = img.shape

        objects = []
        with txt_path.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            parts = line.split(",")
            if len(parts) < 6:
                console.print(
                    f"[yellow] Skipping invalid line {i+1} in {txt_path}[/yellow]"
                )
                continue

            try:
                x_min = int(parts[0])
                y_min = int(parts[1])
                width = int(parts[2])
                height = int(parts[3])
                label_id = parts[5]

                x_max = x_min + width
                y_max = y_min + height

                label_name = self.label_dict.get(label_id, "Unknown")
                if label_name == "Unknown":
                    console.print(
                        f"[yellow] Unknown label ID '{label_id}' in {txt_path}[/yellow]"
                    )

                objects.append(
                    {
                        "name": label_name,
                        "xmin": x_min,
                        "ymin": y_min,
                        "xmax": x_max,
                        "ymax": y_max,
                    }
                )

                if self.draw_bboxes:
                    cv2.rectangle(
                        img, (x_min, y_min), (x_max, y_max), self.color, self.thickness
                    )

            except ValueError as e:
                console.print(
                    f"[red] Error parsing line {i+1} in {txt_path}: {e}[/red]"
                )
                continue

        # Save image
        cv2.imwrite(str(img_output_path), img)

        # Generate and save XML
        xml_content = self.create_voc_xml(
            filename=img_path.name,
            img_path=str(img_path),
            width=w,
            height=h,
            depth=d,
            objects=objects,
        )
        xml_output_path.write_text(xml_content, encoding="utf-8")

    def convert(self):
        """Main conversion method with rich progress tracking."""
        txt_files = list(self.input_ann_folder.glob("*.txt"))
        if not txt_files:
            console.print("[bold red] No .txt annotation files found![/bold red]")
            return

        success_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Converting...", total=len(txt_files))

            for txt_file in txt_files:
                try:
                    img_path = self._find_image_file(txt_file.stem)
                    if not img_path:
                        console.print(
                            f"[yellow] Image not found for {txt_file.stem}[/yellow]"
                        )
                        progress.advance(task)
                        continue

                    xml_output_path = self.output_ann_folder / f"{txt_file.stem}.xml"
                    img_output_path = self.output_img_folder / img_path.name

                    self.process_annotation_file(
                        txt_file, img_path, xml_output_path, img_output_path
                    )
                    success_count += 1

                except Exception as e:
                    console.print(f"[red] Failed processing {txt_file}: {e}[/red]")
                finally:
                    progress.advance(task)

        console.print(
            "[bold green]✅ Done!"
            f" Converted {success_count}/{len(txt_files)} file(s).[/bold green]"
        )


def main(
    input_img_folder: str = "VisDrone2019-DET-train/images",
    input_ann_folder: str = "VisDrone2019-DET-train/annotations",
    output_img_folder: str = "VisDrone2019-DET-train/images_xml",
    output_ann_folder: str = "VisDrone2019-DET-train/annotations_xml",
    draw_bboxes: bool = False,
):
    """Convert VisDrone annotations to Pascal VOC XML format.

    Args:
            input_img_folder (str): Path to input images directory.
            input_ann_folder (str): Path to input annotations (.txt) directory.
            output_img_folder (str): Path to save converted images.
            output_ann_folder (str): Path to save generated XML files.
            draw_bboxes (bool): Whether to draw bounding boxes on output images.
    """
    converter = VisDroneToVOCConverter(
        input_img_folder=input_img_folder,
        input_ann_folder=input_ann_folder,
        output_img_folder=output_img_folder,
        output_ann_folder=output_ann_folder,
        draw_bboxes=draw_bboxes,
    )
    converter.convert()


if __name__ == "__main__":
    fire.Fire(main)
