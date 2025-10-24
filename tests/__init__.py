"""
Tests for VisDrone Toolkit.

Run tests with:
    pytest tests/
    pytest tests/ -v
    pytest tests/ --cov=visdrone_toolkit
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
