"""Batched sampling sweep over an Aria Stage-B checkpoint.

For each (temperature, top-k, min-p) setting in the grid and each TEST
prompt MIDI, sample ``N`` variations and save them as .mid files. The
checkpoint is loaded once and reused across the whole sweep — the
``sample_batch`` helper from ``aria.inference.sample_cuda`` does the
batched generation across the ``N`` variations of a single prompt.

Top-k is not exposed by upstream ``aria generate`` (it only ships min-p
and top-p). We add it as a logits filter applied before ``softmax(/temp)``
so the requested grid is honoured end-to-end without monkey-patching the
upstream package — see the ``_sample_batch_topk`` helper below, which is a
minimal fork of ``sample_batch`` adding a ``top_k`` argument.

Layout:
    $out_dir/sweep/T<t>_k<k>_p<p>/prompt_<i>/var_<j>.mid
    $out_dir/sweep_meta.json   # global config

Usage:
    python -m src.sample_aria_sweep \
        --variant v1 \
        --checkpoint path/to/runs/v1/stageB_final.safetensors \
        --test-midi-dir path/to/data/pijama_raw_hawthorne/test \
        --tokenizer-config "" \
        --model-name medium \
        --out-dir path/to/runs/v1/stageC \
        --n-prompts 4 --n-variations 20 \
        --prompt-duration 8 \
        --length 2048 \
        --temps 0.8,1.0,1.2 \
        --top-ks 0,24 \
        --min-ps 0.035,0.05
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

# Only stdlib at module-import time; torch / aria pulled in inside main().


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True,
                   help="Tag used in sweep_meta.json (e.g. v1, v2, v3, v4).")
    p.add_argument("--checkpoint", required=True,
                   help="Stage-B safetensors checkpoint.")
    p.add_argument("--test-midi-dir", required=True,
                   help="Directory of TEST MIDIs; first --n-prompts will be used.")
    p.add_argument("--tokenizer-config", default="",
                   help="Path to AbsTokenizer config JSON. Empty -> default.")
    p.add_argument("--model-name", default="medium",
                   help="Aria model config name. medium / medium-emb.")
    p.add_argument("--out-dir", required=True,
                   help="Output root. Sweep MIDIs live under $out_dir/sweep/...")
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--n-variations", type=int, default=20)
    p.add_argument("--prompt-duration", type=int, default=8,
                   help="Seconds of the input MIDI to use as prompt.")
    p.add_argument("--length", type=int, default=2048,
                   help="Tokens to generate per variation.")
    p.add_argument("--temps", default="0.8,1.0,1.2",
                   help="Comma-separated temperatures.")
    p.add_argument("--top-ks", default="0,24",
                   help="Comma-separated top-k values (0 = off).")
    p.add_argument("--min-ps", default="0.035,0.05",
                   help="Comma-separated min-p values.")
    p.add_argument("--aria-repo", default="path/to/aria",
                   help="Path to EleutherAI/aria checkout (for `aria.*` imports).")
    p.add_argument("--device", default="cuda",
                   help="Currently only cuda is supported by sample_batch.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="Inference dtype.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip a (setting, prompt) pair if all var_*.mid exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the settings + prompts and exit without sampling.")
    return p.parse_args()


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _setting_dir(t: float, k: int, p: float) -> str:
    # Use 4-sig-fig formatting that round-trips through filenames.
    return f"T{t:g}_k{k:d}_p{p:g}"


def _gather_prompt_paths(test_midi_dir: Path, n: int, seed: int) -> list[Path]:
    paths = sorted(test_midi_dir.glob("*.mid")) + sorted(test_midi_dir.glob("*.midi"))
    if not paths:
        raise FileNotFoundError(f"No .mid/.midi files in {test_midi_dir}")
    # Deterministic shuffle by seed, then take first n.
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    return shuffled[:n]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sweep_root = out_dir / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    temps = _parse_csv_floats(args.temps)
    top_ks = _parse_csv_ints(args.top_ks)
    min_ps = _parse_csv_floats(args.min_ps)

    prompt_paths = _gather_prompt_paths(Path(args.test_midi_dir),
                                        args.n_prompts, args.seed)

    settings = [(t, k, p) for t in temps for k in top_ks for p in min_ps]
    meta = {
        "variant": args.variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_midi_dir": str(Path(args.test_midi_dir).resolve()),
        "tokenizer_config": args.tokenizer_config or "default",
        "model_name": args.model_name,
        "n_prompts": args.n_prompts,
        "n_variations": args.n_variations,
        "prompt_duration_s": args.prompt_duration,
        "length": args.length,
        "temps": temps,
        "top_ks": top_ks,
        "min_ps": min_ps,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "prompt_paths": [str(p) for p in prompt_paths],
        "settings": [
            {"T": t, "top_k": k, "min_p": p, "dir": _setting_dir(t, k, p)}
            for (t, k, p) in settings
        ],
    }
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sweep] {len(settings)} settings x {len(prompt_paths)} prompts x "
          f"{args.n_variations} variations  ({args.variant})")
    print(f"[sweep] meta -> {out_dir / 'sweep_meta.json'}")

    if args.dry_run:
        for (t, k, p) in settings:
            print(f"  setting {_setting_dir(t, k, p)}")
        for pp in prompt_paths:
            print(f"  prompt  {pp}")
        return 0

    # ===== Heavy imports — only after dry-run / arg check. =====
    sys.path.insert(0, args.aria_repo)
    import torch
    from ariautils.tokenizer import AbsTokenizer
    from ariautils.midi import MidiDict
    from aria.inference import get_inference_prompt
    from aria.inference.model_cuda import TransformerLM
    from aria.config import load_model_config
    from aria.model import ModelConfig
    from safetensors.torch import load_file

    # Optional tokenizer cfg.
    if args.tokenizer_config:
        tokenizer = AbsTokenizer(args.tokenizer_config)
    else:
        tokenizer = AbsTokenizer()
    print(f"[sweep] tokenizer vocab={tokenizer.vocab_size} "
          f"include_pedal={tokenizer.config.get('include_pedal', False)}")

    model_config = ModelConfig(**load_model_config(name=args.model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerLM(model_config)
    state_dict = load_file(filename=args.checkpoint)
    # strict=False so a vocab-extended ckpt (v4) still loads even if the
    # state dict doesn't exactly match the configured model.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[sweep] WARN: missing keys: {missing[:5]}... (total {len(missing)})")
    if unexpected:
        print(f"[sweep] WARN: unexpected keys: {unexpected[:5]}... "
              f"(total {len(unexpected)})")
    model = model.cuda().eval()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Pre-tokenise prompts once.
    prompt_tokens: dict[Path, list] = {}
    for pp in prompt_paths:
        try:
            md = MidiDict.from_midi(str(pp))
            tokens = get_inference_prompt(
                midi_dict=md,
                tokenizer=tokenizer,
                prompt_len_ms=1e3 * args.prompt_duration,
            )
            prompt_tokens[pp] = tokens
        except Exception as e:
            print(f"[sweep] WARN: failed to tokenise prompt {pp.name}: {e}; skipping")

    if not prompt_tokens:
        raise SystemExit("No prompts tokenised; cannot proceed.")

    # ===== Sampling loop =====
    t0 = time.time()
    n_done = 0
    n_total = len(settings) * len(prompt_tokens) * args.n_variations

    for (t, k, p) in settings:
        setting_dir = sweep_root / _setting_dir(t, k, p)
        setting_dir.mkdir(parents=True, exist_ok=True)
        for pi, prompt_path in enumerate(prompt_paths):
            if prompt_path not in prompt_tokens:
                continue
            prompt = prompt_tokens[prompt_path]
            prompt_dir = setting_dir / f"prompt_{pi:d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(prompt_dir.glob("var_*.mid"))
            if args.skip_existing and len(existing) >= args.n_variations:
                n_done += args.n_variations
                continue

            # Each (setting, prompt) is one batched generation across
            # n_variations samples — this is how Aria's own sample_batch
            # operates (variations share the prompt).
            try:
                seqs = _sample_batch_topk(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_variations=args.n_variations,
                    max_new_tokens=args.length,
                    temp=t,
                    top_k=k,
                    min_p=p,
                    dtype=dtype,
                )
            except Exception as e:
                print(f"[sweep] ERROR sampling {_setting_dir(t, k, p)} "
                      f"prompt_{pi}: {e}; continuing")
                continue

            # Save each variation as .mid via the tokenizer's detokenize path.
            for vi, seq in enumerate(seqs):
                mid_path = prompt_dir / f"var_{vi:02d}.mid"
                try:
                    md = tokenizer.detokenize(seq)
                    md.to_midi().save(str(mid_path))
                except Exception as e:
                    print(f"[sweep] WARN: detokenize failed for "
                          f"{_setting_dir(t, k, p)}/prompt_{pi}/var_{vi:02d}: {e}")
                    continue
            n_done += args.n_variations
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f"[sweep] {_setting_dir(t, k, p)}/prompt_{pi}: "
                  f"{n_done}/{n_total}  rate={rate:.2f}/s  "
                  f"eta={eta/60:.1f}m")

    print(f"[sweep] done in {(time.time() - t0)/60:.1f} min "
          f"({n_done} / {n_total} samples)")
    return 0


def _sample_batch_topk(
    model,
    tokenizer,
    prompt: list,
    num_variations: int,
    max_new_tokens: int,
    temp: float,
    top_k: int,
    min_p: float,
    dtype,
):
    """Minimal fork of ``aria.inference.sample_cuda.sample_batch`` that
    adds an optional ``top_k`` logits filter applied before
    ``softmax(/temp)``. When ``top_k == 0`` it falls back to plain min-p
    sampling (identical to upstream).

    We do not use ``torch.compile`` here — the latency win is dominated by
    the long-context KV-cache prefill, and compile interacts badly with
    short benchmark sweeps (it recompiles on shape changes).
    """
    import torch
    from aria.inference import sample_min_p
    from aria.inference.sample_cuda import (
        prefill,
        decode_one,
        update_seq_ids_,
    )

    assert 0.0 < temp <= 2.0
    assert 0.0 <= min_p <= 1.0
    assert top_k >= 0

    prompt_len = len(prompt)
    total_len = prompt_len + max_new_tokens
    if total_len > 8192:
        raise ValueError(f"prompt+gen={total_len} exceeds 8192 max_seq_len")

    dim_tok_inserted = [False] * num_variations
    eos_tok_seen = [False] * num_variations

    seq = torch.stack([
        torch.tensor(
            tokenizer.encode(prompt + [tokenizer.pad_tok] * (total_len - prompt_len))
        )
        for _ in range(num_variations)
    ]).cuda()

    model.setup_cache(
        batch_size=num_variations,
        max_seq_len=total_len,
        dtype=dtype,
    )

    with torch.autocast("cuda", dtype=dtype), torch.inference_mode():
        for idx in range(prompt_len, total_len):
            # CUDNN backend gives ~2.6x speedup over MATH on Blackwell sm_121
            # (GB10/B200). EFFICIENT is ~2.5x, MATH was the original default
            # because it's universally supported. The benchmarked CUDNN
            # speedup brought a 14 h sweep down to ~5.5 h. Falls back to MATH
            # implicitly via PyTorch if CUDNN is not available on the host.
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION
            ):
                if idx == prompt_len:
                    logits = prefill(
                        model,
                        idxs=seq[:, :idx],
                        input_pos=torch.arange(0, idx, device=seq.device),
                    )[:, -1]
                else:
                    logits = decode_one(
                        model,
                        idxs=seq[:, idx - 1: idx],
                        input_pos=torch.tensor(
                            [idx - 1],
                            device=seq.device,
                            dtype=torch.int,
                        ),
                    )

            # Optional top-k filter: zero-out everything below the k-th max.
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
                kth = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < kth,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = torch.softmax(logits / temp, dim=-1)
            next_token_ids = sample_min_p(probs, min_p).flatten()

            update_seq_ids_(
                seq=seq,
                idx=idx,
                next_token_ids=next_token_ids,
                dim_tok_inserted=dim_tok_inserted,
                eos_tok_seen=eos_tok_seen,
                max_len=total_len,
                force_end=False,
                tokenizer=tokenizer,
            )

            if all(eos_tok_seen):
                break

    decoded = [tokenizer.decode(s) for s in seq.tolist()]
    decoded = [
        (r[: r.index(tokenizer.eos_tok) + 1] if tokenizer.eos_tok in r else r)
        for r in decoded
    ]
    return decoded


if __name__ == "__main__":
    raise SystemExit(main())
