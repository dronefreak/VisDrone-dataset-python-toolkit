"""
Tests for visualization utilities.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from visdrone_toolkit.visualization import (
    CLASS_COLORS,
    CLASS_NAMES,
    plot_training_curves,
    visualize_annotations,
    visualize_comparison,
    visualize_predictions,
)


class TestVisualizeAnnotations:
    """Tests for visualize_annotations function."""

    def test_basic_visualization(self, sample_image_array, sample_boxes, sample_labels, temp_dir):
        """Test basic annotation visualization."""
        output_path = temp_dir / "test_viz.png"

        fig = visualize_annotations(
            sample_image_array,
            sample_boxes,
            sample_labels,
            save_path=str(output_path),
            show=False,
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_visualization_with_tensors(self, sample_image_array, temp_dir):
        """Test visualization with torch tensors."""
        import torch

        boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        output_path = temp_dir / "test_tensor_viz.png"

        fig = visualize_annotations(
            sample_image_array,
            boxes,
            labels,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_visualization_empty_boxes(self, sample_image_array, temp_dir):
        """Test visualization with no boxes."""
        boxes = np.array([]).reshape(0, 4)
        labels = np.array([], dtype=np.int64)

        output_path = temp_dir / "test_empty_viz.png"

        fig = visualize_annotations(
            sample_image_array,
            boxes,
            labels,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_custom_title(self, sample_image_array, sample_boxes, sample_labels):
        """Test visualization with custom title."""
        fig = visualize_annotations(
            sample_image_array,
            sample_boxes,
            sample_labels,
            title="Custom Title",
            show=False,
        )

        assert "Custom Title" in fig.axes[0].get_title()
        plt.close(fig)


class TestVisualizePredictions:
    """Tests for visualize_predictions function."""

    def test_basic_prediction_visualization(
        self, sample_image_array, sample_boxes, sample_labels, sample_scores, temp_dir
    ):
        """Test basic prediction visualization."""
        output_path = temp_dir / "test_pred_viz.png"

        fig = visualize_predictions(
            sample_image_array,
            sample_boxes,
            sample_labels,
            sample_scores,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_score_threshold_filtering(self, sample_image_array, temp_dir):
        """Test that score threshold filters predictions."""
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        labels = np.array([1, 4], dtype=np.int64)
        scores = np.array([0.9, 0.3], dtype=np.float32)

        output_path = temp_dir / "test_threshold_viz.png"

        # With high threshold, only one box should be displayed
        fig = visualize_predictions(
            sample_image_array,
            boxes,
            labels,
            scores,
            score_threshold=0.5,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_prediction_with_low_scores(
        self, sample_image_array, sample_boxes, sample_labels, temp_dir
    ):
        """Test prediction visualization with low confidence scores."""
        scores = np.array([0.1, 0.2, 0.15], dtype=np.float32)

        output_path = temp_dir / "test_low_score_viz.png"

        fig = visualize_predictions(
            sample_image_array,
            sample_boxes,
            sample_labels,
            scores,
            score_threshold=0.05,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)


class TestVisualizeComparison:
    """Tests for visualize_comparison function."""

    def test_comparison_visualization(
        self, sample_image_array, sample_boxes, sample_labels, sample_scores, temp_dir
    ):
        """Test side-by-side comparison visualization."""
        output_path = temp_dir / "test_comparison_viz.png"

        fig = visualize_comparison(
            sample_image_array,
            sample_boxes,
            sample_labels,
            sample_boxes,  # Use same boxes as predictions
            sample_labels,
            sample_scores,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_comparison_different_detections(self, sample_image_array, temp_dir):
        """Test comparison with different ground truth and predictions."""
        gt_boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        gt_labels = np.array([1], dtype=np.int64)

        pred_boxes = np.array([[95, 95, 205, 205], [300, 300, 400, 400]], dtype=np.float32)
        pred_labels = np.array([1, 4], dtype=np.int64)
        pred_scores = np.array([0.95, 0.87], dtype=np.float32)

        output_path = temp_dir / "test_diff_comparison_viz.png"

        fig = visualize_comparison(
            sample_image_array,
            gt_boxes,
            gt_labels,
            pred_boxes,
            pred_labels,
            pred_scores,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_basic_training_curves(self, temp_dir):
        """Test plotting training curves."""
        train_losses = [2.5, 2.0, 1.5, 1.0, 0.8]
        val_losses = [2.3, 2.1, 1.7, 1.2, 1.0]

        output_path = temp_dir / "test_curves.png"

        fig = plot_training_curves(
            train_losses,
            val_losses,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_training_curves_with_metrics(self, temp_dir):
        """Test plotting with additional metrics."""
        train_losses = [2.5, 2.0, 1.5, 1.0, 0.8]
        val_losses = [2.3, 2.1, 1.7, 1.2, 1.0]
        metrics = {
            "precision": [0.5, 0.6, 0.7, 0.8, 0.85],
            "recall": [0.4, 0.55, 0.65, 0.75, 0.80],
        }

        output_path = temp_dir / "test_curves_metrics.png"

        fig = plot_training_curves(
            train_losses,
            val_losses,
            metrics=metrics,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)

    def test_training_curves_train_only(self, temp_dir):
        """Test plotting with only training losses."""
        train_losses = [2.5, 2.0, 1.5, 1.0, 0.8]

        output_path = temp_dir / "test_curves_train_only.png"

        fig = plot_training_curves(
            train_losses,
            save_path=str(output_path),
            show=False,
        )

        assert output_path.exists()
        plt.close(fig)


class TestConstants:
    """Tests for visualization constants."""

    def test_class_colors_count(self):
        """Test that we have colors for all classes."""
        assert len(CLASS_COLORS) == 12

    def test_class_names_count(self):
        """Test that we have names for all classes."""
        assert len(CLASS_NAMES) == 12

    def test_class_colors_format(self):
        """Test that colors are RGB tuples."""
        for color in CLASS_COLORS.values():
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_class_names_match_dataset(self):
        """Test that class names match dataset classes."""
        from visdrone_toolkit.utils import VISDRONE_CLASSES

        assert CLASS_NAMES == VISDRONE_CLASSES
