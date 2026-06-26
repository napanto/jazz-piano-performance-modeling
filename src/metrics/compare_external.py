"""Compare our OA metrics to values reported in external papers.

This script reads our sweep CSV and plots bars for:
 - Our default (GENERATED_DIR baseline) mean_oa
 - Our best sweep mean_oa
 - External methods (hard-coded from paper 2402.14285v4): SCG OA on Maestro, Muscore, Pop

Usage:
  python -m src.metrics.compare_external \
    --ours-csv runs/balanced/metrics_sweep.csv \
    --out runs/balanced/comparison_scg_vs_ours.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare OA vs external SCG paper")
    p.add_argument("--ours-csv", default="runs/balanced/metrics_sweep.csv")
    p.add_argument("--out", default="runs/balanced/comparison_scg_vs_ours.png")
    return p.parse_args()


def load_ours(csv_path: str) -> tuple[float | None, float | None]:
    df = pd.read_csv(csv_path)
    # Default: pick t==1.0,k==0 if present as "default", else max first matching
    default = None
    cand = df[(df["temperature"].round(2) == 1.0) & (df["top_k"] == 0)]
    if not cand.empty and cand["mean_oa"].notna().any():
        default = float(cand.iloc[0]["mean_oa"])
    best = None
    if df["mean_oa"].notna().any():
        best = float(df.sort_values("mean_oa", ascending=False).iloc[0]["mean_oa"])
    return default, best


def build_df(ours_default: float | None, ours_best: float | None) -> pd.DataFrame:
    rows = []
    if ours_default is not None:
        rows.append({"method": "Ours (default)", "dataset": "PiJAMA", "mean_oa": ours_default})
    if ours_best is not None:
        rows.append({"method": "Ours (best sweep)", "dataset": "PiJAMA", "mean_oa": ours_best})
    # External SCG paper (Huang et al., 2024) Table 2, last column per dataset (approx)
    # NOTE: Datasets differ; this is a qualitative comparison only.
    rows.extend([
        {"method": "SCG (Huang24)", "dataset": "Maestro", "mean_oa": 0.943},
        {"method": "SCG (Huang24)", "dataset": "Muscore", "mean_oa": 0.934},
        {"method": "SCG (Huang24)", "dataset": "Pop", "mean_oa": 0.939},
    ])
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out: str) -> None:
    plt.figure(figsize=(7.5, 4.2))
    ax = sns.barplot(data=df, x="dataset", y="mean_oa", hue="method")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Mean OA (↑)")
    ax.set_xlabel("")
    ax.set_title("Mean OA: Ours vs SCG (Huang et al., 2024) — datasets differ")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h), ha='center', va='bottom', fontsize=8, rotation=0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    ours_default, ours_best = load_ours(args.ours_csv)
    df = build_df(ours_default, ours_best)
    make_plot(df, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

