"""MIDI generation harness for the PerformanceRNN (LSTM) baseline.

Exposes a primer / target-duration / sampling-temperature interface so the
generated pool of clips can be fed back to ``eval_unified.py --generated-dir``
for distributional (OA/KLD) metrics. The loader keys off ``model_type`` in the
checkpoint payload, so additional autoregressive models can be slotted in
without changing the sampling or MIDI-writing code below.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import pretty_midi
import torch

from src.data.tokenizer import (
    PerformanceNote,
    PerformanceTokenizer,
    tokens_to_time_axis,
)
from src.model.performance_rnn import PerformanceRNN, PerformanceRNNConfig
from src.utils.config import apply_yaml_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI clips from a PerformanceRNN checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="outputs/generated")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Number of samples to generate in parallel per forward pass.")
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--primer-json", default=None)
    p.add_argument("--primer-length", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-type", default="auto", choices=["auto", "lstm"])
    p.add_argument("--config", default=None, help="YAML config (uses [generate] section).")
    apply_yaml_defaults(p, section="generate")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device, model_type: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    detected = payload.get("model_type")
    if model_type == "auto":
        if detected:
            model_type = detected
        elif "hidden_size" in (payload.get("config") or {}):
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


def default_primer(tokenizer: PerformanceTokenizer) -> list[int]:
    return [
        tokenizer.velocity_base + 15,
        tokenizer.note_on_base + 60,
        tokenizer.time_shift_base + 30,
        tokenizer.note_off_base + 60,
        tokenizer.time_shift_base + 15,
    ]


def load_primer_tokens(args: argparse.Namespace, tokenizer: PerformanceTokenizer) -> list[int]:
    if args.primer_json:
        payload = json.loads(Path(args.primer_json).read_text())
        tokens = list(payload.get("tokens", []))
    else:
        tokens = default_primer(tokenizer)
    if not tokens:
        tokens = default_primer(tokenizer)
    primer_len = max(1, args.primer_length)
    if len(tokens) >= primer_len:
        return tokens[:primer_len]
    return tokens + [tokenizer.time_shift_base] * (primer_len - len(tokens))


def trim_tokens_by_duration(tokens: Sequence[int], tokenizer: PerformanceTokenizer, seconds: float) -> list[int]:
    if seconds <= 0:
        return list(tokens)
    elapsed = 0.0
    trimmed: list[int] = []
    for token in tokens:
        trimmed.append(token)
        if tokenizer.time_shift_base <= token < tokenizer.velocity_base:
            delta = (token - tokenizer.time_shift_base + 1) * tokenizer.resolution
            elapsed += delta
        if elapsed >= seconds:
            break
    return trimmed


def notes_to_pretty_midi(notes: Sequence[PerformanceNote], tokenizer: PerformanceTokenizer) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="jazz_piano_generated")
    for note in notes:
        start = float(note.start)
        end = max(float(note.end), start + tokenizer.resolution)
        velocity = int(max(1, min(127, note.velocity)))
        instrument.notes.append(pretty_midi.Note(start=start, end=end, pitch=int(note.pitch), velocity=velocity))
    pm.instruments.append(instrument)
    return pm


def determine_event_budget(args: argparse.Namespace, tokenizer: PerformanceTokenizer) -> int:
    if args.max_events:
        return max(1, args.max_events)
    approx_time_tokens = int(math.ceil(args.seconds / tokenizer.resolution))
    return max(1024, approx_time_tokens * 3)


def generate_batch(
    base_idx: int,
    batch_size: int,
    model,
    cfg,
    args: argparse.Namespace,
    tokenizer: PerformanceTokenizer,
    primer_tokens: list[int],
    device: torch.device,
) -> list[list[int]]:
    """Generate ``batch_size`` sequences in parallel. Returns per-sample
    trimmed token lists."""
    torch.manual_seed(args.seed + base_idx)
    primer_row = torch.tensor(primer_tokens, dtype=torch.long, device=device)
    primer_tensor = primer_row.unsqueeze(0).expand(batch_size, -1).contiguous()
    steps = determine_event_budget(args, tokenizer)
    sequence = model.generate(
        primer_tensor, steps=steps,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    return [
        trim_tokens_by_duration(sequence[b].tolist(), tokenizer, args.seconds)
        for b in range(batch_size)
    ]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    tokenizer = PerformanceTokenizer()
    model, model_type, cfg = load_checkpoint(Path(args.checkpoint), device, args.model_type)
    print(f"Loaded {model_type} checkpoint from {args.checkpoint}")
    primer_tokens = load_primer_tokens(args, tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while idx < args.num_samples:
        bsz = min(args.batch_size, args.num_samples - idx)
        trimmed_batch = generate_batch(
            idx, bsz, model, cfg, args, tokenizer, primer_tokens, device,
        )
        for trimmed in trimmed_batch:
            notes = tokenizer.decode(trimmed)
            _, duration = tokens_to_time_axis(trimmed, tokenizer.resolution)
            pm = notes_to_pretty_midi(notes, tokenizer)
            stem = f"sample_{idx:03d}"
            midi_path = output_dir / f"{stem}.mid"
            json_path = output_dir / f"{stem}.json"
            pm.write(str(midi_path))
            payload = {
                "tokens": trimmed,
                "meta": {
                    "model_type": model_type,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "seconds_target": args.seconds,
                    "seed": args.seed + idx,
                    "batch_size": bsz,
                    "primer_source": args.primer_json or "default",
                },
            }
            json_path.write_text(json.dumps(payload, indent=2))
            print(f"Wrote {midi_path} (~{duration:.1f}s, {len(trimmed)} tokens)")
            idx += 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
