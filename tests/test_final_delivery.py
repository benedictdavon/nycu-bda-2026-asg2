import gzip
from pathlib import Path

import pandas as pd
import pytest

from experiments.final_submission.run import permitted_training_origins
from scripts.reproduce_submission import OFFICIAL_ORIGIN, inventory, main, utc_time


def test_official_origin_and_mature_training_schedule():
    origin = pd.Timestamp(OFFICIAL_ORIGIN)
    assert origin.tz_convert("Asia/Taipei").isoformat() == "2026-11-22T00:00:00+08:00"
    origins = permitted_training_origins(origin)
    assert len(origins) == 56
    assert origins[0] == pd.Timestamp("2026-09-19T16:00:00Z")
    assert origins[-1] == pd.Timestamp("2026-11-13T16:00:00Z")
    assert all(value + pd.Timedelta(days=7) <= origin - pd.Timedelta(hours=3) for value in origins)


def test_inventory_sorts_hours_numerically_and_detects_changed_bytes(tmp_path):
    for hour in (10, 2):
        with gzip.open(tmp_path / f"2026-11-01-{hour}.json.gz", "wb") as stream:
            stream.write(b"{}\n")
    rows = inventory(tmp_path)
    assert [row["name"] for row in rows] == ["2026-11-01-2.json.gz", "2026-11-01-10.json.gz"]
    path = tmp_path / rows[0]["name"]
    path.write_bytes(b"different archive bytes")
    assert inventory(tmp_path)[0]["sha256"] != rows[0]["sha256"]


def test_inventory_rejects_duplicate_hour_spellings(tmp_path):
    for name in ("2026-11-01-0.json.gz", "2026-11-01-00.json.gz"):
        (tmp_path / name).write_bytes(b"archive")
    with pytest.raises(ValueError, match="Duplicate UTC"):
        inventory(tmp_path)


def test_historical_csv_requires_rehearsal_marker(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--official-repo", str(tmp_path), "--raw", str(tmp_path),
              "--output-dir", str(tmp_path / "output"), "--student-id", "314561002",
              "--origin", "2026-10-31T16:00:00Z"])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_naive_origin_is_rejected():
    with pytest.raises(ValueError, match="UTC offset"):
        utc_time("2026-11-22T00:00:00")
