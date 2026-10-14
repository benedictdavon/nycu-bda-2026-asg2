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
