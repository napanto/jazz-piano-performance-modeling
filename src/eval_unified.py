"""Evaluation harness for the PerformanceRNN (LSTM) baseline.

Loads a checkpoint and computes:
  * token-level NLL and perplexity on the held-out split
  * (optional) OA/KLD distributional metrics against a directory of generated
    JSON token files

The loader keys off ``model_type`` in the checkpoint payload so additional
autoregressive models can reuse the same NLL / distributional code paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PerformanceDataset
from src.data.tokenizer import PerformanceTokenizer
from src.data.utils import list_token_paths, load_split_ids
from src.metrics.features import FeatureExtractor
from src.metrics.oa_kld import compare_distributions, pairwise_euclidean, pairwise_euclidean_cross
from src.model.performance_rnn import PerformanceRNN, PerformanceRNNConfig
from src.utils.config import apply_yaml_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval (NLL + OA/KLD) for the PerformanceRNN baseline.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processed-root", default="data/processed/pijama")
    p.add_argument("--splits-manifest", default="data/splits/pijama_album_splits.json")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--generated-dir", default=None)
    p.add_argument("--max-target-clips", type=int, default=200)
    p.add_argument("--max-generated-clips", type=int, default=200)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--model-type", default="auto", choices=["auto", "lstm"])
    p.add_argument("--augment-target", action="store_true", default=False,
                   help="Enable random transposition + time-stretch on the test set when "
                        "computing OA/KLD (matches the v1 protocol; inflates target variance and OA). "
                        "Off by default for the cleaner, stricter eval.")
    p.add_argument("--config", default=None)
    apply_yaml_defaults(p, section="eval")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device, model_type: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    detected = payload.get("model_type")
    if model_type == "auto":
        if detected:
            model_type = detected
        elif "config" in payload and "hidden_size" in (payload.get("config") or {}):
            model_type = "lstm"
        else:
            raise ValueError("Could not auto-detect model_type; pass --model-type explicitly")
    if model_type != "lstm":
        raise ValueError(f"Unsupported model_type {model_type}")
    cfg = PerformanceRNNConfig(**payload.get("config", {}))
    model = PerformanceRNN(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, model_type, cfg


@torch.no_grad()
def eval_nll(
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer: PerformanceTokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """Standard token-level cross-entropy NLL on the chosen split."""
    dataset = PerformanceDataset(
        processed_root=Path(args.processed_root),
        split=args.split,
        sequence_length=args.sequence_length,
        clip_seconds=15.0,
        clips_per_track=4,
        splits_manifest=Path(args.splits_manifest),
        tokenizer=tokenizer,
        transpose_min=0,
        transpose_max=0,
        time_stretch_options=(1.0,),  # disable augmentation for clean eval
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    losses_sum = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        losses_sum += float(loss.item())
        total_tokens += labels.numel()
    avg_nll = losses_sum / max(1, total_tokens)
    return {"nll": avg_nll, "perplexity": float(np.exp(avg_nll))}


def gather_feature_vectors(paths: List[Path], extractor: FeatureExtractor) -> Dict[str, list[np.ndarray]]:
    features: Dict[str, list[np.ndarray]] = {}
    for path in paths:
        sample = extractor.from_json(path)
        for name, value in sample.features.items():
            features.setdefault(name, []).append(value.astype(np.float32))
    return features


def eval_distributional(args: argparse.Namespace, tokenizer: PerformanceTokenizer) -> dict:
    if not args.generated_dir:
        return {}
    extractor = FeatureExtractor(tokenizer)
    allowed_ids = load_split_ids(Path(args.splits_manifest), args.split)
    target_paths = list_token_paths(Path(args.processed_root), args.split, allowed_ids)[: args.max_target_clips]
    generated_paths = sorted(Path(args.generated_dir).glob("*.json"))[: args.max_generated_clips]
    target_features = gather_feature_vectors(target_paths, extractor)
    generated_features = gather_feature_vectors(generated_paths, extractor)
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
    return results


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    tokenizer = PerformanceTokenizer()
    model, model_type, cfg = load_checkpoint(Path(args.checkpoint), device, args.model_type)
    print(f"Loaded {model_type} checkpoint from {args.checkpoint}")
    print(f"  config: {cfg}")
    logloss = eval_nll(model, args, tokenizer, device)
    metrics = eval_distributional(args, tokenizer)
    output = {
        "model_type": model_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "logloss": logloss,
        "distribution_metrics": metrics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
