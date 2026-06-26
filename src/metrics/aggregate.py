"""Aggregate evaluation JSONs into a CSV and plots.

Usage:
  python -m src.metrics.aggregate \
    --pattern 'runs/balanced/eval_test_t*_k*.json' \
    --out-csv runs/balanced/metrics_sweep.csv \
    --out-png runs/balanced/metrics_sweep.png
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate eval metrics JSONs into CSV/plot")
    p.add_argument("--pattern", default="runs/balanced/eval_test_t*_k*.json")
    p.add_argument("--out-csv", default="runs/balanced/metrics_sweep.csv")
    p.add_argument("--out-png", default="runs/balanced/metrics_sweep.png")
    p.add_argument("--out-heatmap", default="runs/balanced/metrics_sweep_heatmap.png")
    p.add_argument("--out-pairplot", default="runs/balanced/metrics_sweep_pairplot.png")
    p.add_argument("--pairplot-cols", nargs="*", default=None,
                   help="Optional list of columns for pairplot (defaults to key metrics and a few feature OAs)")
    p.add_argument("--pairplot-hue", default="top_k", help="Hue for pairplot (e.g., 'top_k' or 'temperature')")
    return p.parse_args()


def parse_t_k_from_name(name: str) -> tuple[float | None, int | None]:
    # Accept names like eval_test_t1.2_k24.json
    m = re.search(r"_t([0-9]+(?:\.[0-9]+)?)_k([0-9]+)\.", name)
    if m:
        t = float(m.group(1))
        k = int(m.group(2))
        return t, k
    return None, None


def load_rows(pattern: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        t, k = parse_t_k_from_name(Path(path).name)
        dist = data.get("distribution_metrics", {})
        agg = dist.get("_aggregate", {})
        row = {
            "path": path,
            "temperature": t,
            "top_k": k,
            "mean_oa": agg.get("mean_oa"),
            "nll": data.get("logloss", {}).get("nll"),
            "perplexity": data.get("logloss", {}).get("perplexity"),
        }
        # Flatten per-feature OA/KLD into columns like oa_<feature>, kld_<feature>
        for feat, stats in dist.items():
            if feat == "_aggregate" or not isinstance(stats, dict):
                continue
            oa = stats.get("oa")
            kld = stats.get("kld")
            row[f"oa_{feat}"] = oa
            row[f"kld_{feat}"] = kld
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(7, 4))
    # Keep only rows where t and k are present
    sdf = df.dropna(subset=["temperature", "top_k", "mean_oa"]).copy()
    if sdf.empty:
        # Nothing to plot
        plt.close()
        return
    sns.lineplot(data=sdf, x="temperature", y="mean_oa", hue="top_k", marker="o")
    plt.title("Mean OA vs Temperature (by top-k)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def make_heatmap(df: pd.DataFrame, out_png: str) -> None:
    sdf = df.dropna(subset=["temperature", "top_k", "mean_oa"]).copy()
    if sdf.empty:
        return
    table = sdf.pivot_table(index="top_k", columns="temperature", values="mean_oa", aggfunc="max")
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(table, annot=True, fmt=".3f", cmap="viridis")
    ax.set_title("Mean OA heatmap (top-k × temperature)")
    ax.set_xlabel("temperature")
    ax.set_ylabel("top-k")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def make_pairplot(df: pd.DataFrame, out_png: str, cols: list[str] | None, hue: str) -> None:
    # Default columns: core metrics and representative feature OAs
    default_cols = [
        "mean_oa", "nll", "perplexity",
        "oa_pitch_class_hist", "oa_pitch_interval_hist",
        "oa_velocity_stats", "oa_note_length_hist", "oa_note_density",
        "temperature", "top_k",
    ]
    if cols is None or len(cols) == 0:
        cols = default_cols
    present = [c for c in cols if c in df.columns]
    if len(present) < 3:  # need at least 3 to make the grid useful
        return
    sdf = df[present].dropna()
    if sdf.empty:
        return
    # Ensure hue column exists and is included
    hue_col = hue if hue in sdf.columns else None
    g = sns.pairplot(sdf, vars=[c for c in present if c not in (hue_col,)], hue=hue_col, diag_kind="kde", corner=True)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    g.fig.suptitle("Metrics Pairplot", y=1.02)
    g.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(g.fig)


def main() -> int:
    args = parse_args()
    df = load_rows(args.pattern)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    make_plot(df, args.out_png)
    make_heatmap(df, args.out_heatmap)
    make_pairplot(df, args.out_pairplot, args.pairplot_cols, args.pairplot_hue)

    # Print quick best summary
    if not df.empty and df["mean_oa"].notna().any():
        best = df.sort_values("mean_oa", ascending=False).iloc[0]
        print(f"Best: temp={best['temperature']}, top_k={best['top_k']}, mean_oa={best['mean_oa']:.4f}")
        print(f"CSV -> {args.out_csv}\nPNG -> {args.out_png}\nHEATMAP -> {args.out_heatmap}")
    else:
        print(f"No rows matched pattern {args.pattern}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
