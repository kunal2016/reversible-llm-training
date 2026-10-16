# Reversible Language Models — Training Report


## 1. Objective

Train a ~20M-parameter language model on ~50M tokens and compare a standard
(non-reversible) baseline against reversible variants (Euler / midpoint, plus a
leapfrog variant), measuring the effect of reversibility on peak memory,
throughput, final loss, and the maximum stable batch size, and identifying which
variant performs best.

Reversibility follows **Karami et al., "Reversing Large Language Models for
Efficient Training and Fine-Tuning" (arXiv:2512.02056)**: reversible layers cast
as time-reversible ODE integrators, reconstructing hidden states during backprop
instead of storing them.

---

## 2. Experiment Setup

Identical across all runs so the comparison is clean:

| Item | Value |
|---|---|
| Model params | 20.75M (identical for all 4 configs) |
| Dataset | TinyStories (GPT-2 BPE) |
| Dataset size | 50M tokens (47.5M train / 2.5M val) — the corpus, **not** tokens trained per run |
| Sequence length | 256 |
| Layers / d_model / heads | 10 / 256 / 8 |
| Optimizer / LR | AdamW, 3e-4, cosine decay, 100 warmup |
| Betas / weight decay | (0.9, 0.95) / 0.1 |
| Training steps | 2000 (fixed-batch & max-batch runs in 4.1–4.2) |
| Training tokens / run | batch × seq × steps, e.g. 32×256×2000 = **16.4M** at batch 32 |
| Batch size (fixed runs) | 32 |
| Precision | bf16 autocast |
| Dropout | 0 (required for exact reconstruction) |
| Integrator step h | 0.5 (midpoint uses 2h, leapfrog h²) |
| GPU | NVIDIA T4 (16GB, Colab) |
| Seed | 1337 |

Batch-size sweep method: double the batch from 8 until CUDA OOM, then binary
back-off to the exact ceiling (`sweep.py`).

**Execution environment.** Runs were executed on **Google Colab** with an **NVIDIA
T4 (16 GB)** GPU (the local laptop has only Intel integrated graphics, no CUDA),
PyTorch 2.x + CUDA, bf16 autocast. Data was tokenized once to Google Drive and
reused across all runs. The executed notebook with full logs is
`reversible_llm_colab_run.ipynb`; the clean runner is `reversible_llm_colab.ipynb`.

---

## 3. Run Configurations

| Run | Reversible? | Integrator | Params |
|---|---|---|---|
| A — Baseline | No | — (standard pre-norm) | 20.75M |
| B — Euler | Yes | Hamiltonian coupling (1st order) | 20.75M |
| C — Midpoint | Yes | `p⁺ = p⁻ + 2h·f(p)` | 20.75M |
| D — Leapfrog | Yes | `p⁺ = 2p − p⁻ + h²·f(p)` | 20.75M |

---

## 4. Results

### Fixed batch (32), 2000 steps

| Run | Final train loss | Val loss | Tokens/sec | Peak memory (GB) |
|---|---|---|---|---|
| A — Baseline | 2.8213 | 2.7235 | 14,230 | 6.819 |
| B — Euler | 2.8370 | 2.7312 | 12,962 | 5.258 |
| C — Midpoint | **2.7669** | **2.6632** | 12,830 | 5.258 |
| D — Leapfrog | 2.8094 | 2.7017 | 12,689 | 5.258 |

### Max stable batch (OOM sweep)

| Run | Max stable batch | Peak memory at max (GB) | Tokens/sec at max |
|---|---|---|---|
| A — Baseline | 56 | 11.699 | 14,190 |
| B — Euler | 64 | 10.221 | 13,200 |
| C — Midpoint | 64 | 10.221 | 13,003 |

### Trained at max sustainable batch (all-participants item)

Full 2000-step runs at each config's max stable batch, recording final loss +
tokens/sec + peak memory at that batch.

| Run | Batch | Final train loss | Val loss | Tokens/sec | Peak memory (GB) |
|---|---|---|---|---|---|
| A — Baseline | 56 | 2.5358 | 2.5715 | 12,651 | 11.698 |
| B — Euler | 64 | 2.4885 | 2.5471 | 11,517 | 10.221 |
| C — Midpoint | 64 | **2.4109** | **2.4753** | 12,261 | 10.221 |

At each config's max batch the reversible variants show lower loss and lower peak
memory — but **this comparison is not token-matched, so it must not be read as an
accuracy gain from reversibility.** Each run used a fixed 2000 *steps*, so the arms
trained on different token totals: baseline 56×256×2000 = **28.7M**, reversible
64×256×2000 = **32.8M** (~14% more). The fixed-batch runs in 4.1 likewise trained on
32×256×2000 = **16.4M** tokens — not the 50M dataset size. The lower reversible loss
here is partly just more data seen; the memory and max-batch findings stand, the
accuracy comparison does not.

**Correction (mentor feedback).** To attribute any accuracy change to the method, every
arm must train to the *same* token budget. `equal_token_budget_experiment.ipynb` reruns
all arms to an identical **50M-token budget** (`--target_tokens 50000000`, which sets
iters per batch: 32→6104, 56→3488, 64→3052). **Warmup scales to 5% of steps** (305/174/153); peak LR is held at 3e-4 (cosine → 10%) identically across all arms, so coupling is the only variable. Corrected numbers below.

### Equal token-budget runs (50M tokens/arm) — token-matched comparison

Every arm trained to **train_tokens = 50,003,968** (iters auto-set per batch), warmup 5%, LR 3e-4.

**Experiment A — equal batch (32), 50M tokens** (controlled: batch *and* tokens matched)

| Run | Batch | Train loss | Val loss | Tokens/sec | Peak mem (GB) |
|---|---|---|---|---|---|
| Baseline | 32 | 2.1087 | 2.0887 | 13,959 | 6.819 |
| Euler | 32 | 2.1327 | 2.1022 | 11,887 | 5.258 |
| **Midpoint** | 32 | **2.0675** | **2.0520** | 11,827 | 5.258 |
| Leapfrog | 32 | 2.1095 | 2.0858 | 11,657 | 5.258 |

**Experiment B — max batch per arm, 50M tokens** (token-fair version of the max-batch comparison)

| Run | Batch | Train loss | Val loss | Tokens/sec | Peak mem (GB) |
|---|---|---|---|---|---|
| Baseline | 56 | 2.1946 | 2.2175 | 13,184 | 11.698 |
| Euler | 64 | 2.2072 | 2.2728 | 12,034 | 10.221 |
| **Midpoint** | 64 | **2.1415** | **2.2060** | 11,378 | 10.221 |

![Equal-budget results](results_equal_figure.png)

---

## 5. Analysis

**Memory.** At a fixed batch of 32, all three reversible variants used **5.258 GB
vs the baseline's 6.819 GB — a 22.9% reduction**. Fitting a per-sample memory
line across the fixed-batch and max-batch points gives 0.155 GB/sample for
reversible vs 0.203 GB/sample for baseline: reversible activations cost **23.7%
less per sample**, consistent with the fixed-batch figure. The saving is real and
reproducible but modest, because at only 10 layers the recomputed activations are
a limited share of total memory (weights, optimizer state, embeddings and the
50257-row output projection are fixed overhead that reversibility doesn't touch).

**Max batch size.** Reversible fit **batch 64 vs the baseline's 56 (1.14×)**.
This is far below the paper's ~10×, and the reason is depth: the paper's headline
used 96 layers, where activation memory dominates and removing it frees most of
the budget. At 10 layers the fixed overhead dilutes the gain. Notably reversible
at batch 64 (10.22 GB) still used *less* memory than the baseline at batch 56
(11.70 GB) — it had headroom the baseline lacked.

**Throughput.** At fixed batch the reversible variants were **9–11% slower**
(≈12.7–13.0k vs 14.2k tok/s) — the cost of the extra forward evaluation each
backward pass. Because the batch-size gain here is small, that saved memory did
not convert into a throughput win (the mechanism behind the paper's ~2× speed-up).
At this depth reversibility is a memory technique, not a speed technique.

**Euler vs midpoint (token-matched — the authoritative accuracy result).** Under
the corrected equal-budget comparison (Experiment A: same batch 32 *and* same 50M
tokens), **midpoint is still the best variant** — val **2.0520**, below the
baseline's 2.0887 (−0.037). Leapfrog is essentially tied with baseline (2.0858,
−0.003) and Euler is marginally worse (2.1022, +0.014). Ordering: **midpoint <
leapfrog ≈ baseline < euler**. Once the token confound is removed, the gap is
smaller than the earlier fixed-step numbers suggested, but the *direction* holds:
reversibility costs no accuracy, and the symmetric midpoint integrator modestly
helps. Experiment B (each arm at its max batch, still 50M tokens) tells the same
story — midpoint edges baseline (2.2060 vs 2.2175), euler trails — and its losses
are uniformly higher than Experiment A because a larger batch under a fixed token
budget means fewer optimizer steps (3052–3488 vs 6104): a batch-size/steps
tradeoff, not a reversibility effect. All arms trained stably at h = 0.5, LR 3e-4.

(The earlier fixed-iteration runs in Sections 4.1–4.2 are retained for the memory,
throughput and max-batch findings, which are unaffected; their loss comparison is
superseded by the token-matched numbers here.)

**Surprises / gotchas.** (1) All reversible variants report identical peak memory
because the memory is dominated by the recompute footprint, which is the same
across schemes. (2) The OOM ceiling is slightly fragmentation-dependent, so the
batch *ratio* is the robust number, not the absolute value. (3) Dropout had to be
0 — a random mask would break exact reconstruction.

---

## 6. Conclusions

- **Best variant: midpoint** — lowest train and val loss under the token-matched
  comparison (val **2.0520** vs baseline 2.0887, −0.037), the same 22.9% memory
  saving as the other reversible schemes, and stable training. It gives the best
  loss *and* the memory benefit even when every arm sees an identical 50M tokens.
- **Reversibility for the upcoming B200 training targets: yes, with scale.** At
  10 layers the memory win is modest (23%) and doesn't yet pay off in throughput.
  But the benefit grows with depth — the paper's ~10× batch and ~2× throughput
  come at 96 layers — so for the deep models planned on B200 hardware, reversible
  layers (midpoint) are the right choice: they enable much larger batches within
  a fixed memory budget at no loss in quality, and turn that into real throughput
  once activation memory dominates.

---

## 7. Appendix

### Commands used
```bash
python data.py --tokens 50_000_000
python train.py --coupling none     --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling euler    --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling midpoint --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling leapfrog --batch_size 32 --max_iters 2000 --eval_val

# Equal 50M-token budget (correction): iters auto-set per batch
python train.py --coupling none     --batch_size 32 --target_tokens 50_000_000 --eval_val
python train.py --coupling none     --batch_size 56 --target_tokens 50_000_000 --eval_val
python train.py --coupling midpoint --batch_size 64 --target_tokens 50_000_000 --eval_val
python sweep.py --coupling none
python sweep.py --coupling euler
python sweep.py --coupling midpoint
```

### Key config
10 layers, d_model 256, 8 heads, mlp_ratio 4, seq_len 256, GPT-2 BPE (vocab 50257),
AdamW lr 3e-4 cosine, weight decay 0.1, bf16, dropout 0, integrator step h = 0.5,
seed 1337, NVIDIA T4.

### Correctness (test_reversible.py)
Inverses exact to ~1e-7; reversible custom-backward gradients match plain autograd
to ~1e-8; all four configs have identical parameter counts; reversible forward
stores no intermediate autograd graph.
