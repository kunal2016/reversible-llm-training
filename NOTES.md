# Implementation Notes — Reversible LM

Three mechanisms worth understanding (and good material for the report's method
section): the custom autograd Function, why the coupling is exactly invertible,
and how the batch-size sweep finds the memory ceiling.

---

## 1. The custom `torch.autograd.Function`

Normally PyTorch records a graph of every intermediate tensor during the forward
pass so it can backprop automatically. **That graph is the stored activations**,
and it grows with depth. To avoid storing them we opt out of automatic recording
and supply our own backward via `torch.autograd.Function` — PyTorch then treats
the whole stack as one opaque op.

```python
class _ReversibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, blocks):
        ctx.blocks = blocks
        u, v = torch.chunk(x, 2, dim=-1)
        with torch.no_grad():                 # <-- no graph built here
            for blk in blocks:
                u, v = blk(u, v)
        out = torch.cat([u, v], dim=-1)
        ctx.save_for_backward(out)            # store ONLY the final output
        return out
```

- **`torch.no_grad()`** — inside this block no graph is built, so every layer's
  intermediate `(u, v)` is computed and immediately freed. This is what turns
  activation memory from "grows with depth" into "constant."
- **`ctx`** carries data from forward to backward: `save_for_backward(out)` keeps
  the single final tensor; `ctx.blocks` keeps the modules so backward can re-run
  them.
- **The return** looks like an ordinary tensor; PyTorch attaches a `grad_fn`
  pointing at our `backward`, so `loss.backward()` calls our code.

```python
    @staticmethod
    def backward(ctx, grad_out):
        out, = ctx.saved_tensors
        u, v = torch.chunk(out, 2, dim=-1)
        du, dv = torch.chunk(grad_out, 2, dim=-1)
        for blk in reversed(ctx.blocks):      # walk DOWN the stack
            u, v, du, dv = blk.backward_pass(u, v, du, dv)
        return torch.cat([du, dv], dim=-1), None
```

The `backward` contract: receive `grad_out` (gradient of loss w.r.t. this op's
output), return the gradient w.r.t. each input of `forward`. `forward` took
`(x, blocks)`, so we return `(grad_x, None)` — `None` because `blocks` is not a
differentiable input. Each `backward_pass` also fills the block parameters'
`.grad` as a side effect. Crucially, backward starts from the one saved tensor
`out` and **reconstructs** each layer's activations on the way down, so nothing
in between ever had to be stored. It iterates `reversed(blocks)` because we saved
the top of the stack and peel back to the input.

---

## 2. Why the coupling is exactly invertible

Pure algebra — no learned inverse, no approximation. The `euler`/Hamiltonian
block, forward:

```
q = q_prev + F(p_prev)      ... (1)
p = p_prev + G(q)           ... (2)
```

We keep **both** outputs `q` and `p`. Given `(q, p)`, recover `(q_prev, p_prev)`:

```
p_prev = p − G(q)           ... invert (2):  G(q) needs only q, which we have
q_prev = q − F(p_prev)      ... invert (1):  F(p_prev) needs p_prev, just recovered
```

```python
def inverse(self, u2, v2):        # u2 = q, v2 = p
    p_prev = v2 - self.G(u2)      # subtract, do NOT invert G
    q_prev = u2 - self.F(p_prev)
    return q_prev, p_prev
```

The key point: **we never invert the neural nets `F` or `G`** (that would be
intractable). We *re-evaluate them forward* and subtract. It works only because
each function is fed an argument we already know at that step — `G` takes `q`
(kept), `F` takes `p_prev` (just recovered). Order matters: undo (2) before (1),
because (2)'s inverse produces the `p_prev` that (1)'s inverse needs.

A **standard** block `p = p_prev + f(p_prev)` cannot be reversed: recovering
`p_prev` needs `f(p_prev)`, which needs `p_prev`. Circular → must store it. The
reversible trick breaks the circularity by evaluating `f` at a state the backward
pass already holds (the other stream, or the middle layer in midpoint).

Midpoint is the same idea, one line:

```
forward:  p_next = p_prev + 2h·f(p_cur)     # state (p_prev, p_cur)
inverse:  p_prev = p_next − 2h·f(p_cur)     # f at p_cur, which we keep
```

**This is why dropout must be 0.** `inverse` re-runs `F` / `G` / `f`. A random
dropout mask would differ between the forward pass and the reconstruction, so the
subtraction wouldn't cancel to the true previous state and gradients would be
wrong. Determinism is a correctness requirement.

---

## 3. How the batch-size sweep finds the ceiling

The "ceiling" is the largest batch that fits in GPU memory without a CUDA
out-of-memory (OOM) error. It can't be computed analytically (depends on GPU,
model, fragmentation), so `sweep.py` finds it empirically: **exponential search
then binary back-off**, catching OOM as a normal exception.

```python
def try_batch(base_args, batch):
    args.batch_size = batch
    args.max_iters = base_args.probe_iters   # ~60 iters — enough to hit peak memory
    try:
        res = run(args)
        torch.cuda.empty_cache()
        return res                           # fit succeeded
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()         # release the partial allocation
            return None                      # does not fit
        raise                                # a real error — don't swallow it
```

We actually *run* a few steps rather than just build the model, because peak
memory occurs during the **backward pass** (gradients + optimizer state + the
largest activations all live at once). Only a real step exercises that peak.

**Phase 1 — double until it breaks:**
```python
batch = start_batch          # e.g. 8
while True:
    res = try_batch(args, batch)
    if res is None: break     # OOM: true ceiling is in (best, batch]
    best = batch
    batch *= 2                # 8, 16, 32, 64, ...
```

**Phase 2 — binary search the gap:**
```python
hi = best * 2                        # known-bad upper bound
probe = best + (hi - best) // 2
while probe > best:
    if try_batch(args, probe) is not None:
        best = probe                 # fit -> raise floor
    else:
        hi = probe                   # OOM -> lower ceiling
    probe = best + (hi - best) // 2
```

Final `best` is the reported max stable batch; throughput and peak memory at that
batch go in the results table.

**Caveats for the report:** memory fragmentation makes the exact ceiling slightly
noisy (96 one run, 92 another); the `empty_cache()` calls matter so a failed
allocation doesn't crowd the next probe. Report the **ratio** between configs
(reversible vs baseline), not the absolute batch — that's the robust number, and
it's how the paper frames its ~10× result across different GPUs.
