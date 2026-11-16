"""Regression checks for relocating runs and renaming serialized model classes."""
import sys

import joblib

from src.baseline_models import ModelBundle
from src.model_io import load_model_bundle
from src.paths import resolve_run_path


def test_original_manifest_paths_resolve_to_relocated_runs(tmp_path):
    current = tmp_path / "experiments/exp005_temporal_dataset/runs/build-synthetic"
    current.mkdir(parents=True)
    original = tmp_path / "experiments/datasets/exp005/build-synthetic"
    assert resolve_run_path(original) == current
    assert resolve_run_path(current) == current
    old_models = tmp_path / "experiments/models/exp006/run-synthetic"
    new_models = tmp_path / "experiments/exp006_baseline_comparison/runs/run-synthetic"
    new_models.mkdir(parents=True)
    assert resolve_run_path(old_models) == new_models


def test_existing_paths_and_unrelated_paths_are_preserved(tmp_path):
    original = tmp_path / "experiments/datasets/exp005/build-existing"
    original.mkdir(parents=True)
    assert resolve_run_path(original) == original
    unrelated = tmp_path / "other/run"
    assert resolve_run_path(unrelated) == unrelated


def test_recorded_local_paths_resolve_after_workspace_cleanup(tmp_path, monkeypatch):
    from src import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    assert paths.resolve_run_path(tmp_path / "assignment2/data/db") == tmp_path / "data/legacy/assignment2/data/db"
    assert paths.resolve_run_path(tmp_path / "assignment2_backup/snapshot") == tmp_path / "data/legacy/snapshots/snapshot"
    assert paths.resolve_run_path(tmp_path / "assignment2_new/revision_smoke/assignment2/scripts/evaluate.py") == tmp_path / "data/course/scripts/evaluate.py"
    assert paths.resolve_run_path(tmp_path / "requirements-exp011.txt") == tmp_path / "configs/environments/requirements-exp011.txt"
    old = tmp_path / "assignment2"
    old.mkdir()
    assert paths.resolve_run_path(old) == old


def test_pre_rename_model_bundle_can_be_loaded(tmp_path, monkeypatch):
    from src import baseline_models

    monkeypatch.setitem(sys.modules, "src.exp006_models", baseline_models)
    monkeypatch.setattr(ModelBundle, "__module__", "src.exp006_models")
    bundle = ModelBundle("zero", ["feature"], {}, {}, {"push": 0.5})
    path = tmp_path / "original_bundle.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(ModelBundle, "__module__", "src.baseline_models")
    monkeypatch.delitem(sys.modules, "src.exp006_models")
    loaded = load_model_bundle(path)
    assert isinstance(loaded, ModelBundle)
    assert loaded.kind == "zero"
    assert loaded.thresholds == {"push": 0.5}
