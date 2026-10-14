"""Resolve archived artifact paths after the descriptive experiment-folder migration."""
from __future__ import annotations

from pathlib import Path


def resolve_run_path(path: str | Path) -> Path:
    """Prefer an existing path; otherwise translate the two original run layouts.

    Historical manifests remain immutable, including their original absolute paths.
    Only these known directory sequences are translated; no files are moved here.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        return resolved
    parts = resolved.parts
    replacements = {
        ("experiments", "datasets", "exp005"): ("experiments", "exp005_temporal_dataset", "runs"),
        ("experiments", "models", "exp006"): ("experiments", "exp006_baseline_comparison", "runs"),
    }
    for old, new in replacements.items():
        for i in range(len(parts) - len(old) + 1):
            if parts[i:i + len(old)] == old:
                return Path(*parts[:i], *new, *parts[i + len(old):])
    return resolved
