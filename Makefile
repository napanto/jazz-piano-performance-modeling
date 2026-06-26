# Makefile for the PerformanceRNN (LSTM) baseline harness.
#
# Generation and evaluation for the LSTM baseline go through the shared
# src.generate_unified / src.eval_unified harnesses. The Aria fine-tune
# pipelines are driven by their own orchestrators under scripts/.
#
# Override LSTM_CKPT to point at the trained checkpoint, e.g.
#   make eval-lstm LSTM_CKPT=path/to/best.pt

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

# Trained PerformanceRNN checkpoint (override on the command line or via env).
LSTM_CKPT ?= ../project3/runs/balanced/best.pt

LSTM_RUN := runs/lstm_rerun
LSTM_SAMPLES := outputs/lstm_samples

N_SAMPLES := 64
CLIP_SECONDS := 30

# Run python inside the project venv. Override PYRUN for a different launcher.
PYRUN ?= source .venv/bin/activate && python -u

# ---------------------------------------------------------------------------
# Generation (shared sampling budget)
# ---------------------------------------------------------------------------
.PHONY: gen-lstm
gen-lstm:
	$(PYRUN) -m src.generate_unified --checkpoint $(LSTM_CKPT) \
	  --num-samples $(N_SAMPLES) --seconds $(CLIP_SECONDS) \
	  --output-dir $(LSTM_SAMPLES) --temperature 1.0

# ---------------------------------------------------------------------------
# Evaluation: NLL + OA/KLD on the test split vs the generated pool
# ---------------------------------------------------------------------------
.PHONY: eval-lstm
eval-lstm:
	$(PYRUN) -m src.eval_unified --checkpoint $(LSTM_CKPT) \
	  --generated-dir $(LSTM_SAMPLES) \
	  --output $(LSTM_RUN)/eval_test.json

# ---------------------------------------------------------------------------
# Sampling sweep across temperatures and top-k
# ---------------------------------------------------------------------------
.PHONY: sweep-lstm
sweep-lstm:
	@for t in 0.8 1.0 1.2; do \
	  for k in 0 24; do \
	    out=$(LSTM_SAMPLES)_t$${t}_k$${k}; \
	    $(PYRUN) -m src.generate_unified --checkpoint $(LSTM_CKPT) \
	      --num-samples $(N_SAMPLES) --seconds $(CLIP_SECONDS) \
	      --output-dir $$out --temperature $$t --top-k $$k; \
	    $(PYRUN) -m src.eval_unified --checkpoint $(LSTM_CKPT) \
	      --generated-dir $$out \
	      --output $(LSTM_RUN)/eval_test_t$${t}_k$${k}.json; \
	  done; \
	done

# ---------------------------------------------------------------------------
# Environment smoke test
# ---------------------------------------------------------------------------
.PHONY: check-env
check-env:
	$(PYRUN) scripts/check_env.py
