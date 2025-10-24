# Tests

Comprehensive unit tests for VisDrone Toolkit using pytest.

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=visdrone_toolkit --cov-report=term-missing

# Run specific test file
pytest tests/test_dataset.py

# Run specific test class
pytest tests/test_dataset.py::TestVisDroneDataset

# Run specific test
pytest tests/test_dataset.py::TestVisDroneDataset::test_dataset_initialization
```

### Using Makefile

```bash
# Run tests with coverage
make test

# Run tests in parallel (faster)
pytest tests/ -n auto
```

## Test Structure

```bash
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures (shared test data)
├── test_dataset.py             # Dataset tests
├── test_utils.py               # Utility function tests
├── test_visualization.py       # Visualization tests
├── test_converters.py          # Converter tests
└── README.md                   # This file
```

## Test Coverage

### test_dataset.py

Tests for `VisDroneDataset` class:

- ✅ Dataset initialization
- ✅ Loading images and annotations
- ✅ Parsing VisDrone annotation format
- ✅ Filtering ignored boxes
- ✅ Filtering crowd regions
- ✅ Target dictionary format
- ✅ Edge cases (empty annotations, invalid paths)

### test_utils.py

Tests for utility functions:

- ✅ Model factory (`get_model`)
- ✅ Collate function for DataLoader
- ✅ Box IoU computation
- ✅ Metrics computation (precision, recall, F1)
- ✅ Constants validation

### test_visualization.py

Tests for visualization utilities:

- ✅ Annotation visualization
- ✅ Prediction visualization
- ✅ Side-by-side comparison
- ✅ Training curves plotting
- ✅ Score threshold filtering
- ✅ Edge cases (empty boxes, low scores)

### test_converters.py

Tests for annotation converters:

- ✅ VisDrone → COCO conversion
- ✅ VisDrone → YOLO conversion
- ✅ Format validation
- ✅ Filtering options
- ✅ Edge cases (missing files, malformed data)

## Fixtures

Reusable test data defined in `conftest.py`:

### Directories

- `temp_dir` - Temporary directory (auto-cleaned)
- `mock_visdrone_dataset` - Complete mock VisDrone dataset

### Images

- `sample_image` - PIL Image (640x480 RGB)
- `sample_image_array` - Numpy array (640x480 RGB)

### Annotations

- `sample_boxes` - Bounding boxes [x1, y1, x2, y2]
- `sample_labels` - Class labels
- `sample_scores` - Confidence scores
- `sample_target` - PyTorch target dict
- `sample_prediction` - PyTorch prediction dict

### Others

- `device` - PyTorch device (CPU for testing)
- `num_classes` - Number of classes (12)
- `class_names` - VisDrone class names

## Writing New Tests

### Basic Test Structure

```python
import pytest
from visdrone_toolkit import VisDroneDataset

class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        expected = 42

        # Act
        result = my_function()

        # Assert
        assert result == expected

    def test_with_fixture(self, sample_image):
        """Test using a fixture."""
        # Fixtures are automatically provided
        assert sample_image is not None
```

### Using Fixtures

```python
def test_with_temp_dir(self, temp_dir):
    """Create temporary files for testing."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")
    assert test_file.exists()
    # temp_dir is automatically cleaned up after test

def test_with_mock_dataset(self, mock_visdrone_dataset):
    """Use mock VisDrone dataset."""
    dataset = VisDroneDataset(
        image_dir=str(mock_visdrone_dataset['image_dir']),
        annotation_dir=str(mock_visdrone_dataset['annotation_dir']),
    )
    assert len(dataset) == mock_visdrone_dataset['num_images']
```

### Testing Exceptions

```python
def test_invalid_input(self):
    """Test error handling."""
    with pytest.raises(ValueError):
        invalid_function()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("model_name", [
    "fasterrcnn_resnet50",
    "fasterrcnn_mobilenet",
    "fcos_resnet50",
    "retinanet_resnet50",
])
def test_all_models(self, model_name):
    """Test all model types."""
    model = get_model(model_name, num_classes=12, pretrained=False)
    assert model is not None
```

## Test Markers

### Skip Tests

```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature(self):
    pass

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_feature(self):
    pass
```

### Mark as Slow

```python
@pytest.mark.slow
def test_long_running(self):
    """This test takes a while."""
    pass

# Run without slow tests:
# pytest tests/ -m "not slow"
```

## Coverage Requirements

Target coverage: **>80%** for all modules

Check coverage:

```bash
pytest tests/ --cov=visdrone_toolkit --cov-report=html
# Open htmlcov/index.html in browser
```

## Continuous Integration

Tests run automatically on:

- Every push to main branch
- Every pull request
- Before releases

See `.github/workflows/ci.yml` for CI configuration.

## Best Practices

### 1. Test Organization

```python
class TestFeatureGroup:
    """Group related tests in classes."""

    def test_basic_case(self):
        """Test the basic case."""
        pass

    def test_edge_case(self):
        """Test edge cases."""
        pass
```

### 2. Clear Test Names

```python
# Good
def test_dataset_returns_correct_number_of_items(self):
    pass

# Bad
def test_dataset(self):
    pass
```

### 3. AAA Pattern

```python
def test_something(self):
    # Arrange - Set up test data
    input_data = [1, 2, 3]

    # Act - Execute the code
    result = process(input_data)

    # Assert - Verify results
    assert result == [2, 4, 6]
```

### 4. Use Fixtures for Setup

```python
# Don't repeat setup in every test
@pytest.fixture
def prepared_dataset():
    return VisDroneDataset(...)

def test_with_dataset(self, prepared_dataset):
    assert len(prepared_dataset) > 0
```

### 5. Test One Thing

```python
# Good - Tests one specific thing
def test_filter_removes_ignored_boxes(self):
    dataset = VisDroneDataset(..., filter_ignored=True)
    _, target = dataset[0]
    # Check only ignored boxes are filtered

# Bad - Tests multiple things
def test_everything(self):
    # Tests filtering AND loading AND conversion...
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest tests/ -v --tb=short
```

### Print Statements

```python
def test_something(self):
    result = calculate()
    print(f"Result: {result}")  # Will show if test fails
    assert result == expected
```

### Drop into Debugger

```bash
pytest tests/ --pdb  # Drop into pdb on failure
```

### Run Last Failed

```bash
pytest tests/ --lf  # Run only tests that failed last time
```

## Performance

### Parallel Execution

```bash
# Install plugin
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
```

### Profile Slow Tests

```bash
pytest tests/ --durations=10  # Show 10 slowest tests
```

## Maintenance

### Update Test Data

If you change the VisDrone format or add features:

1. Update fixtures in `conftest.py`
2. Add new test cases
3. Update this README

### Keep Tests Fast

- Use mocks for expensive operations
- Create minimal test data
- Run slow tests separately

### Review Coverage

```bash
# Check what's not covered
pytest tests/ --cov=visdrone_toolkit --cov-report=term-missing
```

## Troubleshooting

### Import Errors

```bash
# Install package in development mode
pip install -e .
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -r requirements-dev.txt
```

### Matplotlib Issues

```bash
# Tests use non-interactive backend
# If you see display errors, check that matplotlib.use('Agg') is set
```

## Contributing

When contributing:

1. **Write tests** for new features
2. **Update existing tests** if changing behavior
3. **Ensure all tests pass**: `pytest tests/`
4. **Check coverage**: `pytest tests/ --cov=visdrone_toolkit`
5. **Follow test naming conventions**

See `CONTRIBUTING.md` for more details.
