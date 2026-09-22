# Reversible Language Models: Memory-Efficient Training with Reversible Integrators

A from-scratch PyTorch study of a standard transformer LM against **reversible**
variants (Euler / midpoint / leapfrog) — the **midpoint** variant matches baseline
quality at **~23% lower activation memory**.

Reversible layers are cast as time-reversible ODE integrators (after **Karami et
al., arXiv:2512.02056**): hidden states are *reconstructed* during backprop rather
than stored, so activation memory is nearly independent of depth — at the cost of
one extra forward evaluation per layer.

## Results

20.75M params · 10 layers · d_model 256 · 50M tokens (TinyStories, GPT-2 BPE) · NVIDIA T4.

| Config | Val loss | Peak mem @ b32 (GB) | Tokens/sec @ b32 | Max stable batch |
|---|---|---|---|---|
| Baseline | 2.7235 | 6.819 | 14,230 | 56 |
| Euler | 2.7312 | 5.258 | 12,962 | 64 |
| **Midpoint** | **2.6632** | **5.258** | 12,830 | **64** |
| Leapfrog | 2.7017 | 5.258 | 12,689 | — |

![Results](results_figure.png)

- **Memory:** reversible variants use **22.9% less** peak memory at fixed batch
  (5.26 vs 6.82 GB); ~23.7% less activation memory per sample.
- **Quality:** midpoint reaches the **lowest loss of all four — below the
  baseline** — confirming reversibility costs no accuracy and can slightly help.
- **Throughput:** 9–11% slower at fixed batch (the extra recompute). At 10 layers
  the batch-size gain (1.14×) is too small to convert into a throughput win; the
  effect **grows with depth** (the paper reaches ~10× batch / ~2× throughput at 96
  layers). This is a conservative, correct demonstration of the mechanism.

Full write-up in [`REPORT.md`](REPORT.md); method deep-dives in [`NOTES.md`](NOTES.md).

## Files

```
model.py             GPT with baseline + euler/midpoint/leapfrog + reversible autograd
data.py              tokenize ~50M tokens -> train.bin / val.bin (GPT-2 BPE)
train.py             training loop, logs metrics to results.jsonl
sweep.py             doubles batch until OOM -> reports max stable batch
test_reversible.py   correctness checks (invertibility + grad equivalence)
reversible_llm_colab.ipynb   Drive-backed Colab runner (GPU)
```

## The integrators

`f(p)` is one layer's function (Attn + MLP, pre-norm). Two coupled full-width
streams are carried; all configs have the **same parameter count** as the baseline.

| `--coupling` | Forward rule | Reverse (backward reconstruction) |
|---|---|---|
| `none` | `p = p_prev + f(p_prev)` (standard) | not reversible — stores activations |
| `euler` | `q = q_prev + Attn(LN p_prev)`;  `p = p_prev + MLP(LN q)` | `p_prev = p − MLP(LN q)`;  `q_prev = q − Attn(LN p_prev)` |
| `midpoint` | `p⁺ = p⁻ + 2h·f(p)` | `p⁻ = p⁺ − 2h·f(p)` |
| `leapfrog` | `p⁺ = 2p − p⁻ + h²·f(p)` | `p⁻ = 2p − p⁺ + h²·f(p)` |

The block is evaluated at the **intermediate** state `p` (which the backward pass
already holds); a standard block evaluates `f` at the state it's trying to recover,
so it can't be inverted without storing the input. `h` is the step size (`--h`,
default 0.5); the scheme is only marginally stable, so tune `h` if training diverges.

## Reproduce

```bash
pip install -r requirements.txt
python test_reversible.py                       # inverses exact, grads match autograd
python data.py --tokens 50_000_000              # -> data/train.bin, val.bin

python train.py --coupling none     --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling euler    --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling midpoint --batch_size 32 --max_iters 2000 --eval_val
python train.py --coupling leapfrog --batch_size 32 --max_iters 2000 --eval_val

python sweep.py --coupling none                 # max-batch (OOM) sweep
python sweep.py --coupling euler
python sweep.py --coupling midpoint
```

No GPU locally? Use `reversible_llm_colab.ipynb` (set runtime to GPU, Run all).
Keep `--max_iters` and `--seq_len` identical across runs — only change `--coupling`.

### Execution environment

All results in this repo were produced on **Google Colab, NVIDIA T4 (16 GB)**,
PyTorch 2.x + CUDA, bf16 autocast. Data: **50M tokens** of TinyStories via GPT-2
BPE (47.5M train / 2.5M val). The executed notebook with full run logs (GPU check,
correctness checks, per-run loss + `RESULT` lines, OOM sweep, results table) is
[`reversible_llm_colab_run.ipynb`](reversible_llm_colab_run.ipynb);
[`reversible_llm_colab.ipynb`](reversible_llm_colab.ipynb) is the clean,
output-free runner for re-execution.

## Constraints
- **Dropout must be 0** for reversible modes — the backward recomputes `f`, and a
  random dropout mask would make the reconstruction differ from the forward pass.
  The code enforces this when `coupling != none`.
- **Marginal stability** — the step size `h` sits in a narrow stable range.
- `peak_mem_gb` is only meaningful on CUDA (`torch.cuda.max_memory_allocated`).

## Reference
Karami et al., *Reversing Large Language Models for Efficient Training and
Fine-Tuning*, arXiv:2512.02056.
