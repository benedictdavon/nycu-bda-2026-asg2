"""Load trusted local model bundles, including models saved before module renaming."""
from __future__ import annotations

from pathlib import Path
import sys

import joblib

from . import baseline_models


def load_model_bundle(path: str | Path) -> baseline_models.ModelBundle:
    """Load only trusted local files: joblib/pickle loading can execute Python code.

    Older bundles stored the class path src.exp006_models.ModelBundle. Register
    its renamed module at load time so archived models need no rewriting/refitting.
    """
    sys.modules.setdefault("src.exp006_models", baseline_models)
    return joblib.load(path)
