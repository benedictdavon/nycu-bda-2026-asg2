"""Resolve archived artifact paths after the descriptive experiment-folder migration."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Translate local paths recorded before the November workspace cleanup.
# Historical manifests keep their original contents and hashes.
WORKSPACE_MOVES = {
    "assignment2": "data/legacy/assignment2",
    "assignment2_backup": "data/legacy/snapshots",
    "assignment2_new": "data/research",
    "report": "reports/presentation",
    "requirements-exp011.txt": "configs/environments/requirements-exp011.txt",
    "requirements-exp015.txt": "configs/environments/requirements-exp015.txt",
}


def resolve_run_path(path: str | Path) -> Path:
    """Prefer an existing path; otherwise translate recorded workspace/run paths.

    Historical manifests remain immutable, including their original absolute paths.
    Only these known directory sequences are translated; no files are moved here.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        return resolved
    if resolved.is_relative_to(PROJECT_ROOT):
        relative = resolved.relative_to(PROJECT_ROOT)
        if relative.parts and relative.parts[0] in WORKSPACE_MOVES:
            # The revised course checkout was extracted from the old research tree.
            if relative.parts[:3] == ("assignment2_new", "revision_smoke", "assignment2"):
                return PROJECT_ROOT / "data/course" / Path(*relative.parts[3:])
            return PROJECT_ROOT / WORKSPACE_MOVES[relative.parts[0]] / Path(*relative.parts[1:])
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
