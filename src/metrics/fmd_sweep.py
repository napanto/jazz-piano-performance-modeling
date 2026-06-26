"""Compute FMD across all our sample directories and aggregate.

Loops through ``outputs/<model>_*`` dirs that contain >= ``--min-files``
MIDIs, computes FMD against the shared PiJAMA reference, writes a parallel
``runs/<model_or_run>/eval_<dirname>_fmd.json`` per dir, and prints a
summary table.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


_PATTERN = re.compile(
    r"^(?P<model>lstm|transformer|diffusion|latent_diffusion|nmt|nmt_v2|smdim|smdim_v2)"
    r"(?:_(?P<rest>.+))?$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--reference-dir", default="data/fmd_reference/test")
    p.add_argument("--min-files", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-csv", default="reports/fmd_sweep.csv")
    p.add_argument("--only", nargs="*", default=None,
                   help="If set, only process directories whose name contains "
                        "any of these substrings.")
    p.add_argument("--skip-cached", action="store_true", default=True,
                   help="Skip directories where an eval_*_fmd.json already exists.")
    return p.parse_args()


def _gather_dirs(root: Path, min_files: int, only: list[str] | None) -> list[Path]:
    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if only and not any(s in child.name for s in only):
            continue
        n_midis = sum(1 for _ in child.glob("*.mid")) + sum(1 for _ in child.glob("*.midi"))
        if n_midis >= min_files:
            out.append(child)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.outputs_root)
    ref_dir = Path(args.reference_dir)
    dirs = _gather_dirs(root, args.min_files, args.only)
    print(f"Found {len(dirs)} sample dirs with >= {args.min_files} MIDIs.")
    # Lazy-import so this script doesn't fail at import time on machines
    # without CLaMP-2 available.
    from frechet_music_distance import FrechetMusicDistance
    fmd_engine = FrechetMusicDistance(feature_extractor="clamp2", verbose=False)
    rows = []
    for d in dirs:
        out_json = d.parent.parent / "reports" / f"fmd_{d.name}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_cached and out_json.exists():
            row = json.loads(out_json.read_text())
            print(f"  [cached] {d.name}  FMD={row['fmd']:.3f}")
            rows.append({"dir": d.name, **{k: row[k] for k in row if k != "scores"}})
            continue
        # Quick sanity check.
        n_gen = sum(1 for _ in d.glob("*.mid")) + sum(1 for _ in d.glob("*.midi"))
        print(f"  computing FMD for {d.name} (n={n_gen})...")
        try:
            score = fmd_engine.score(reference_path=str(ref_dir), test_path=str(d))
        except Exception as e:
            print(f"    !! failed: {e}")
            continue
        row = {
            "dir": d.name,
            "fmd": float(score),
            "n_generated": n_gen,
            "n_reference": sum(1 for _ in ref_dir.glob("*.midi")),
            "reference_dir": str(ref_dir.resolve()),
        }
        out_json.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"    FMD = {score:.4f}  ->  {out_json}")

    if rows:
        df = pd.DataFrame(rows).sort_values("fmd")
        csv_path = Path(args.results_csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSorted summary (lower is better) -> {csv_path}")
        print(df[["dir", "fmd"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
