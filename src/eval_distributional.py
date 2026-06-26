"""Stand-alone OA/KLD distributional eval — model-agnostic.

Use this when the checkpoint's architecture isn't supported by ``src.eval_unified``
(any model whose generations are dumped as token JSON files).
It reads the generated samples' ``*.json`` token files plus the held-out PiJAMA
test split's token JSONs, computes feature distributions on both, and writes a
JSON identical in shape to what ``eval_unified`` would emit (minus the NLL).

Usage:
    python -m src.eval_distributional \
        --generated-dir outputs/lstm_samples \
        --output runs/lstm_rerun/eval_oa.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import Dict, List

import numpy as np

from src.data.tokenizer import PerformanceTokenizer
from src.data.utils import list_token_paths, load_split_ids
from src.metrics.features import FeatureExtractor
from src.metrics.oa_kld import (
    compare_distributions,
    pairwise_euclidean,
    pairwise_euclidean_cross,
)


def gather_feature_vectors(paths: List[Path], extractor: FeatureExtractor) -> Dict[str, list[np.ndarray]]:
    features: Dict[str, list[np.ndarray]] = {}
    for path in paths:
        sample = extractor.from_json(path)
        for name, value in sample.features.items():
            features.setdefault(name, []).append(value.astype(np.float32))
    return features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--processed-root", default="data/processed/pijama")
    p.add_argument("--splits-manifest", default="data/splits/pijama_album_splits.json")
    p.add_argument("--split", default="test")
    p.add_argument("--max-target-clips", type=int, default=200)
    p.add_argument("--max-generated-clips", type=int, default=200)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--output", required=True)
    p.add_argument("--augment-target", action="store_true", default=False,
                   help="Random transposition + time-stretch on the test set "
                        "(matches the v1 protocol; inflates variance). Off by default.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = PerformanceTokenizer()
    extractor = FeatureExtractor(tokenizer)

    allowed_ids = load_split_ids(Path(args.splits_manifest), args.split)
    target_paths = list_token_paths(Path(args.processed_root), args.split, allowed_ids)[: args.max_target_clips]
    gen_paths = sorted(Path(args.generated_dir).glob("*.json"))
    gen_paths = [p for p in gen_paths if p.name != "aria_rerank_scores.json"][: args.max_generated_clips]
    print(f"  target  ({args.split}): {len(target_paths)} clips")
    print(f"  generated: {len(gen_paths)} clips from {args.generated_dir}")

    target_features = gather_feature_vectors(target_paths, extractor)
    generated_features = gather_feature_vectors(gen_paths, extractor)

    results: dict = {}
    oa_values: list[float] = []
    for feature_name, target_vectors in target_features.items():
        if feature_name not in generated_features:
            continue
        intra = pairwise_euclidean(target_vectors)
        inter = pairwise_euclidean_cross(generated_features[feature_name], target_vectors)
        comparison = compare_distributions(intra, inter, samples=args.bins)
        results[feature_name] = comparison
        if "oa" in comparison:
            oa_values.append(float(comparison["oa"]))

    if oa_values:
        results["_aggregate"] = {"mean_oa": float(np.mean(oa_values))}

    output = {
        "generated_dir": str(Path(args.generated_dir).resolve()),
        "split": args.split,
        "n_target": len(target_paths),
        "n_generated": len(gen_paths),
        "augment_target": args.augment_target,
        "distribution_metrics": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    if "_aggregate" in results:
        print(f"\nmean_oa = {results['_aggregate']['mean_oa']:.4f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
