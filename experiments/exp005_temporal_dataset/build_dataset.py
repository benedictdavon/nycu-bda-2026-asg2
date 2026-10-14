"""EXP-005: build temporal datasets, verify example labels, or build inference features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.data_loader import OfficialData, locate_data_dir
from src.dataset_builder import build_historical, build_inference, dataset_run_dir, load_config
from src.labels import verify_sample_truth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("build", "sample-check", "inference"), required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/temporal_dataset.json"), help="Path to JSON configuration")
    p.add_argument("--project-root", default=".", help="Your PERSONAL project root, not the course clone")
    p.add_argument("--data-dir", default=None, help="Path containing instructor *.parquet")
    p.add_argument("--output-dir", default=None, help="Base directory; run-specific subfolder created")
    p.add_argument("--overwrite", action="store_true", help="Explicitly overwrite generated files for this run ID")
    p.add_argument("--unseal-holdout", action="store_true", help="Only after freezing model selection and holdout rules")
    p.add_argument("--users-csv", default=None, help="Official test_input.csv or another one-column cohort UserID CSV")
    p.add_argument("--cutoff", default="2026-11-11T16:00:00Z", help="Inference cutoff, UTC/offset-aware")
    p.add_argument("--sample-start", default="2026-09-27T13:00:00Z")
    p.add_argument("--sample-as-of", default="2026-10-09T12:00:00Z")
    p.add_argument("--require-perfect", action="store_true", help="Exit nonzero if any example truth label differs")
    args = p.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    cfg = load_config(args.config)
    d = locate_data_dir(root, args.data_dir)
    src = OfficialData(d)
    if args.mode == "sample-check":
        summary = verify_sample_truth(src, start=args.sample_start, label_as_of=args.sample_as_of)
        print(json.dumps(summary, indent=2, default=str))
        if args.require_perfect and summary["total_mismatches"]:
            return 2
        return 0
    if args.mode == "build":
        out = dataset_run_dir(root, src, cfg, "build", args.output_dir)
        result = build_historical(src, cfg, output_dir=out,
                                  unseal_holdout=args.unseal_holdout, overwrite=args.overwrite)
    else:
        out = dataset_run_dir(root, src, cfg, "inference", args.output_dir)
        result = build_inference(src, cfg, output_dir=out, cutoff=args.cutoff,
                                 users_csv=args.users_csv, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"EXP-005 error: {exc}", file=sys.stderr)
        sys.exit(1)
