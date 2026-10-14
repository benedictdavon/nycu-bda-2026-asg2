"""Optional REAL Parquet/TA evaluator regression; not part of offline smoke tests.

Enable from your personal project root (PowerShell):
  $env:RUN_ASG2_LIVE_SAMPLE = '1'
  python -m pytest tests/test_live_official_regression.py -q

Requires the complete official clone and PyArrow. Strict match is anchored to the
calibrated Oct 14 course snapshot, not a guarantee for retroactively revised releases.
"""
import os
from pathlib import Path

import pytest

from src.data_loader import OfficialData, locate_data_dir
from src.labels import verify_sample_truth


@pytest.mark.skipif(os.getenv("RUN_ASG2_LIVE_SAMPLE") != "1",
                    reason="Opt-in only: set RUN_ASG2_LIVE_SAMPLE=1")
def test_exact_official_sample_reconstruction():
    data = OfficialData(locate_data_dir())
    result = verify_sample_truth(data)
    assert result["sample_users"] == 4785
    assert result["binary_decisions"] == 28710
    assert result["total_mismatches"] == 0, (
        "Current snapshot differs from the calibrated sample—inspect target-wise "
        f"mismatches and record the data SHA: {result['mismatches']}"
    )
    assert result["scores"]["score"] == 1.0
