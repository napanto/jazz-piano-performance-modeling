# Expressive Jazz Piano Performance Generation

Generative modelling of expressive solo jazz piano performance on the
[PiJAMA](https://transactions.ismir.net/articles/10.5334/tismir.162) dataset.
A PerformanceRNN LSTM baseline is pretrained on a large piano corpus and
fine-tuned on PiJAMA, and the publicly released Aria piano transformer is
fine-tuned on the same data. Models are compared under a shared distributional
evaluation protocol (Overlapping Area, OA, and KL divergence over symbolic
features, plus Frechet Music Distance, FMD) on an album-aware PiJAMA test split.

## Models and headline results

Two model families. The headline metric is the mean OA (higher is better) of
generated performances against the PiJAMA test split, under each model's best
sampling configuration.

| Model | Description | Mean OA |
|---|---|---|
| **PerformanceRNN (LSTM)** | Stacked-LSTM event-token baseline (Oore et al., 2018), kong split with pedal tokenization | 0.768 |
| **Aria fine-tuned, full quality** | Pretrained autoregressive piano transformer fine-tuned on PiJAMA | **0.911** |
| **Aria fine-tuned, real time** | Smaller real-time variant (drop-in weights for the real-time studio) | 0.804 |

## Repository layout

```
README.md
Makefile               LSTM baseline harness targets
requirements.txt
data/                  symlink to the processed PiJAMA dataset (not in git)
src/
  model/               performance_rnn.py
  data/                event/PerTok tokenizers, dataset, splits, PiJAMA download
  metrics/             OA/KLD, FMD, symbolic feature extraction
  sample_aria_sweep.py eval_aria_metrics.py
  generate_unified.py  eval_unified.py        PerformanceRNN baseline harness
scripts/               Aria pipeline orchestrators, env setup, tokenization
runs/lstm_rerun/       LSTM baseline evaluation results
reports/               published Aria sampling sweep and hardware notes
```

## Public models and demo

- **Trained model weights:** https://huggingface.co/napanto/jazz-piano-performance-modeling
- **Real-time studio** (interactive Aria performance app):
  https://github.com/napanto/aria-realtime-studio

## Environment

The code runs on both ROCm (AMD) and CUDA (NVIDIA) GPUs. Each setup script
creates a `.venv/`, installs dependencies, and runs the environment smoke test
(`scripts/check_env.py`):

```bash
bash scripts/setup_rocm.sh      # AMD / ROCm (inherits the system torch)
bash scripts/setup_cuda.sh      # NVIDIA / CUDA (installs a CUDA-wheel torch)
```

The processed PiJAMA cache is expected under `data/` (a symlink to the dataset
directory); it is not checked into git. Runs record the git SHA, resolved
configuration, and metrics for reproducibility.

## Coding assistants

Parts of this codebase were developed with the help of LLM-based coding
assistants, used under the direction and review of the author.

## License

The code in this repository is released under the Apache License 2.0 (see
`LICENSE`). It is original work; the Aria package (Apache-2.0) and the
`frechet-music-distance` / CLaMP-2 metric (MIT) are used as released
dependencies, not vendored. The trained model weights are distributed
separately under non-commercial terms (CC-BY-NC-SA-4.0), reflecting the PiJAMA
and Aria-MIDI dataset licenses — see the model repository for details.

## References

1. Oore, Simon, Dieleman, Eck, Simonyan. *This Time with Feeling: Learning Expressive Musical Performance.* arXiv:1808.03715 (2018).
2. Yang and Lerch. *On the evaluation of generative models in music.* Neural Computing and Applications (2020).
3. Edwards, Dixon, Benetos. *PiJAMA: Piano Jazz with Automatic MIDI Annotations.* TISMIR (2023).
4. EleutherAI. *Aria*, an autoregressive symbolic piano model. https://github.com/EleutherAI/aria
