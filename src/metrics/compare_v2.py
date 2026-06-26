"""Aggregate v2 evaluation JSONs into a single CSV + comparison plots.

Reads one or more ``eval_unified.py`` output files (one per model family) and
produces:

* ``reports/comparison_v2.csv`` — one row per model with NLL, perplexity,
  mean OA, and per-feature OA/KLD.
* ``reports/comparison_v2.png`` — grouped bar chart of mean OA + per-feature
  OAs across models.
* ``reports/comparison_v2_nll.png`` — bar chart of NLL/perplexity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lstm", required=False, default=None,
                   help="Eval JSON for the LSTM baseline.")
    p.add_argument("--transformer", required=False, default=None,
                   help="Eval JSON for the Music Transformer.")
    p.add_argument("--diffusion", required=False, default=None,
                   help="Eval JSON for the absorbing-diffusion model.")
    p.add_argument("--extra", action="append", default=[],
                   help="Additional eval JSONs, format LABEL=path")
    p.add_argument("--out-csv", default="reports/comparison_v2.csv")
    p.add_argument("--out-png", default="reports/comparison_v2.png")
    p.add_argument("--out-nll", default="reports/comparison_v2_nll.png")
    return p.parse_args()


def load_row(label: str, path: Path) -> dict:
    payload = json.loads(path.read_text())
    row: dict = {"model": label, "path": str(path)}
    ll = payload.get("logloss", {})
    row["nll"] = ll.get("nll")
    row["perplexity"] = ll.get("perplexity")
    row["nll_note"] = ll.get("note", "")
    dist = payload.get("distribution_metrics", {})
    agg = dist.get("_aggregate", {})
    row["mean_oa"] = agg.get("mean_oa")
    for feat, stats in dist.items():
        if feat == "_aggregate" or not isinstance(stats, dict):
            continue
        if "oa" in stats:
            row[f"oa_{feat}"] = stats["oa"]
        if "kld" in stats:
            row[f"kld_{feat}"] = stats["kld"]
    return row


def collect(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    if args.lstm and Path(args.lstm).exists():
        rows.append(load_row("LSTM (v1)", Path(args.lstm)))
    if args.transformer and Path(args.transformer).exists():
        rows.append(load_row("Music Transformer", Path(args.transformer)))
    if args.diffusion and Path(args.diffusion).exists():
        rows.append(load_row("Diffusion", Path(args.diffusion)))
    for extra in args.extra:
        if "=" not in extra:
            raise ValueError(f"--extra expected LABEL=path, got {extra}")
        label, path = extra.split("=", 1)
        if Path(path).exists():
            rows.append(load_row(label, Path(path)))
    return pd.DataFrame(rows)


def plot_oa(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    oa_cols = [c for c in df.columns if c.startswith("oa_")]
    if not oa_cols:
        return
    feats = [c[len("oa_"):] for c in oa_cols]
    long_rows = []
    for _, r in df.iterrows():
        for feat, col in zip(feats, oa_cols):
            if pd.notna(r.get(col)):
                long_rows.append({"model": r["model"], "feature": feat, "OA": r[col]})
        if pd.notna(r.get("mean_oa")):
            long_rows.append({"model": r["model"], "feature": "_aggregate", "OA": r["mean_oa"]})
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return
    plt.figure(figsize=(13, 5.5))
    ax = sns.barplot(data=long_df, x="feature", y="OA", hue="model")
    ax.set_title("Distributional similarity (Yang & Lerch OA, higher is better)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_nll(df: pd.DataFrame, out_png: str) -> None:
    if df.empty or df["nll"].isna().all():
        return
    sub = df.dropna(subset=["nll"]).copy()
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=sub, x="model", y="nll", ax=ax1, color="steelblue")
    ax1.set_title("Test NLL (lower is better)")
    ax1.set_ylabel("nats / token")
    sns.barplot(data=sub, x="model", y="perplexity", ax=ax2, color="darkorange")
    ax2.set_title("Test perplexity (lower is better)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    df = collect(args)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}")
    plot_oa(df, args.out_png)
    print(f"Wrote OA plot: {args.out_png}")
    plot_nll(df, args.out_nll)
    print(f"Wrote NLL plot: {args.out_nll}")
    # Pretty-print a small summary.
    headline = df[["model", "nll", "perplexity", "mean_oa"]].copy()
    print(headline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
