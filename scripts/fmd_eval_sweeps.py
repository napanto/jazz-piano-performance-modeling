"""Compute FMD for the four headline sweeps (Aria v1/v2, LSTM kong/hawthorne).

For each (sweep, config) pair:
  - Flatten generated MIDIs (nested under prompt_*/var_*.mid for Aria;
    samples/sample_*.mid for LSTM) into a temp dir per-config.
  - Compute FMD against the relevant reference pool (kong or hawthorne).
  - Append rows to sweep_summary_fmd.csv next to the original CSV.

Designed to be RESUMABLE: if the per-config result JSON exists,
it's reused instead of recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[1]))
HAWTHORNE_REF = REPO / "data" / "fmd_reference" / "test"          # 141 MIDIs
KONG_REF = REPO / "data" / "fmd_reference" / "test_kong"          # 278 MIDIs

SWEEPS = [
    {
        "name": "aria_v1",
        "kind": "aria",
        "sweep_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v1/stageC/sweep",
        "ref_dir": HAWTHORNE_REF,
        "summary_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v1/stageC/sweep_summary.csv",
        "out_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v1/stageC/sweep_summary_fmd.csv",
        "cache_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v1/stageC/fmd_cache",
    },
    {
        "name": "aria_v2",
        "kind": "aria",
        "sweep_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v2/stageC/sweep",
        "ref_dir": KONG_REF,
        "summary_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v2/stageC/sweep_summary.csv",
        "out_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v2/stageC/sweep_summary_fmd.csv",
        "cache_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v2/stageC/fmd_cache",
    },
    {
        "name": "aria_v6",
        "kind": "aria",
        "sweep_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v6/stageC/sweep",
        "ref_dir": KONG_REF,
        "summary_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v6/stageC/sweep_summary.csv",
        "out_csv": REPO / "reports/aria_sweep_midis/aria_finetune/v6/stageC/sweep_summary_fmd.csv",
        "cache_dir": REPO / "reports/aria_sweep_midis/aria_finetune/v6/stageC/fmd_cache",
    },
    {
        "name": "lstm_kong",
        "kind": "lstm",
        "sweep_dir": REPO / "runs/fs/lstm_pijama_kong_ft_A/stageC/sweep",
        "ref_dir": KONG_REF,
        "summary_csv": REPO / "runs/fs/lstm_pijama_kong_ft_A/stageC/sweep_summary.csv",
        "out_csv": REPO / "runs/fs/lstm_pijama_kong_ft_A/stageC/sweep_summary_fmd.csv",
        "cache_dir": REPO / "runs/fs/lstm_pijama_kong_ft_A/stageC/fmd_cache",
    },
    {
        "name": "lstm_hawthorne",
        "kind": "lstm",
        "sweep_dir": REPO / "runs/fs/lstm_pijama_hawthorne_ft_A/stageC/sweep",
        "ref_dir": HAWTHORNE_REF,
        "summary_csv": REPO / "runs/fs/lstm_pijama_hawthorne_ft_A/stageC/sweep_summary.csv",
        "out_csv": REPO / "runs/fs/lstm_pijama_hawthorne_ft_A/stageC/sweep_summary_fmd.csv",
        "cache_dir": REPO / "runs/fs/lstm_pijama_hawthorne_ft_A/stageC/fmd_cache",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Sweep names to run (default: all). e.g. --only aria_v1 lstm_kong",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config subdir names to evaluate (default: all). e.g. --configs T1.2_k0_p0.035",
    )
    p.add_argument("--backbone", default="clamp2", choices=["clamp2", "clamp"])
    return p.parse_args()


def gather_aria_config_midis(config_dir: Path) -> list[Path]:
    midis = sorted(config_dir.glob("prompt_*/var_*.mid"))
    return midis


def gather_lstm_config_midis(config_dir: Path) -> list[Path]:
    midis = sorted((config_dir / "samples").glob("sample_*.mid"))
    return midis


def parse_config_name(name: str) -> tuple[float, int, float]:
    # T<float>_k<int>_p<float>
    t_s, k_s, p_s = name.split("_")
    T = float(t_s[1:])
    k = int(k_s[1:])
    p = float(p_s[1:])
    return T, k, p


def run_fmd_for_config(
    config_dir: Path,
    ref_dir: Path,
    kind: str,
    cache_dir: Path,
    fmd_engine,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json = cache_dir / f"{config_dir.name}.json"
    if cache_json.exists():
        try:
            row = json.loads(cache_json.read_text())
            print(f"  [cached] {config_dir.name}  FMD={row['fmd']:.4f}")
            return row
        except Exception:
            pass
    if kind == "aria":
        midis = gather_aria_config_midis(config_dir)
    else:
        midis = gather_lstm_config_midis(config_dir)
    if not midis:
        print(f"  [skip] {config_dir.name}: no MIDIs")
        return {"config": config_dir.name, "fmd": float("nan"), "n_gen": 0}

    # Flatten into tmpdir so FMD library sees a flat folder.
    with tempfile.TemporaryDirectory(prefix=f"fmd_{config_dir.name}_") as tmp:
        tmp_path = Path(tmp)
        for i, src in enumerate(midis):
            dst = tmp_path / f"{i:04d}{src.suffix}"
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        try:
            score = fmd_engine.score(reference_path=str(ref_dir), test_path=str(tmp_path))
        except Exception as e:
            print(f"  [FAIL] {config_dir.name}: {e}")
            traceback.print_exc()
            return {"config": config_dir.name, "fmd": float("nan"), "n_gen": len(midis), "error": str(e)}
    row = {
        "config": config_dir.name,
        "fmd": float(score),
        "n_gen": len(midis),
        "n_ref": sum(1 for _ in ref_dir.iterdir() if not _.is_dir()),
        "ref_dir": str(ref_dir),
    }
    cache_json.write_text(json.dumps(row, indent=2))
    print(f"  {config_dir.name}: FMD={score:.4f}  n_gen={len(midis)}")
    return row


def process_sweep(spec: dict, configs_filter: list[str] | None, fmd_engine) -> pd.DataFrame:
    sweep_dir: Path = spec["sweep_dir"]
    ref_dir: Path = spec["ref_dir"]
    print(f"\n===== sweep: {spec['name']} =====")
    print(f"  sweep_dir = {sweep_dir}")
    print(f"  ref_dir   = {ref_dir}")
    if not sweep_dir.exists():
        print("  !! sweep dir missing, skipping")
        return pd.DataFrame()
    if not ref_dir.exists():
        print("  !! ref dir missing, skipping")
        return pd.DataFrame()

    configs = sorted([p for p in sweep_dir.iterdir() if p.is_dir()])
    if configs_filter:
        configs = [c for c in configs if c.name in configs_filter]
    print(f"  {len(configs)} configs to process")

    rows = []
    for cfg in configs:
        try:
            T, k, p = parse_config_name(cfg.name)
        except Exception:
            print(f"  [skip] cannot parse config name {cfg.name}")
            continue
        result = run_fmd_for_config(cfg, ref_dir, spec["kind"], spec["cache_dir"], fmd_engine)
        row = {
            "sweep": spec["name"],
            "config": cfg.name,
            "T": T,
            "top_k": k,
            "min_p": p,
            "fmd": result.get("fmd", float("nan")),
            "n_gen": result.get("n_gen", 0),
            "n_ref": result.get("n_ref", 0),
        }
        rows.append(row)
        # Incremental write so report-writing process sees partial progress.
        df = pd.DataFrame(rows).sort_values(["T", "top_k", "min_p"])
        df.to_csv(spec["out_csv"], index=False)
    print(f"  wrote {spec['out_csv']}")
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    from frechet_music_distance import FrechetMusicDistance
    print(f"Loading FMD engine (backbone={args.backbone})...")
    fmd_engine = FrechetMusicDistance(feature_extractor=args.backbone, verbose=False)
    print("FMD engine ready.")

    selected = [s for s in SWEEPS if (args.only is None or s["name"] in args.only)]
    print(f"Will process sweeps: {[s['name'] for s in selected]}")

    all_rows = []
    for spec in selected:
        df = process_sweep(spec, args.configs, fmd_engine)
        all_rows.append(df)

    print("\n===== HEADLINE: best FMD per sweep =====")
    for df in all_rows:
        if df.empty:
            continue
        df2 = df.dropna(subset=["fmd"]).sort_values("fmd")
        if df2.empty:
            print(f"  {df['sweep'].iloc[0]}: no successful FMD")
            continue
        best = df2.iloc[0]
        print(f"  {best['sweep']}: best FMD={best['fmd']:.4f}  config={best['config']}  (T={best['T']}, k={best['top_k']}, p={best['min_p']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
