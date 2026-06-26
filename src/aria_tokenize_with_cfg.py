"""Build a PretrainingDataset (Aria's pretokenized cache) with a custom
AbsTokenizer config.

Aria's built-in `aria pretrain-dataset` CLI hardcodes the default
tokenizer constructor (`AbsTokenizer()`). For our pipeline we need to
swap in:
  - the demo config (vocab 2 675) for v2
  - the include-pedal-True default (vocab 17 729) for v4

This script wraps `PretrainingDataset.build()` with a tokenizer that we
explicitly construct from a config path.

Usage:
  python3 src/aria_tokenize_with_cfg.py \\
    --load-path data/jsonl/kong_train.jsonl \\
    --save-dir data/v4/tok_stageA/train \\
    --num-epochs 4 \\
    --seq-len 4096 \\
    --tokenizer-config path/to/pedal_only_cfg.json

If --tokenizer-config is empty or omitted, the default AbsTokenizer is
used (vocab 17 727).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--load-path", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--tokenizer-config", default=None,
                   help="Path to AbsTokenizer config JSON. Default: stock.")
    p.add_argument("--aria-repo", default="path/to/aria",
                   help="Path to EleutherAI/aria checkout (for `aria.*` imports)")
    args = p.parse_args()

    sys.path.insert(0, args.aria_repo)
    from ariautils.tokenizer import AbsTokenizer
    from aria.datasets import PretrainingDataset

    if args.tokenizer_config:
        tok = AbsTokenizer(args.tokenizer_config)
    else:
        tok = AbsTokenizer()

    print(f"[tokenize] config={args.tokenizer_config or 'default'}  "
          f"vocab_size={tok.vocab_size}  "
          f"include_pedal={tok.config.get('include_pedal', False)}")

    PretrainingDataset.build(
        tokenizer=tok,
        save_dir=args.save_dir,
        max_seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        midi_dataset_path=args.load_path,
        separate_sequences=False,
        file_embeddings=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
