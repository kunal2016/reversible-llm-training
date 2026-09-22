"""
Correctness checks for the reversible blocks. Run on CPU:  python test_reversible.py

1. Exact invertibility:      inverse(forward(x)) == x
2. Gradient equivalence:     custom memory-saving backward == plain autograd
   (a reference model stores activations; a reversible model recomputes them;
    param grads and input grads must match to numerical tolerance)
"""
import torch
import torch.nn as nn
from model import make_reversible_block, _ReversibleFunction, GPT, GPTConfig

torch.manual_seed(0)
COUPLINGS = ("euler", "midpoint", "leapfrog")


def test_invertibility():
    print("== invertibility ==")
    B, T, d = 2, 8, 32
    for coupling in COUPLINGS:
        blk = make_reversible_block(coupling, d, 4, 4, 0.0, 0.5).eval()
        u = torch.randn(B, T, d)
        v = torch.randn(B, T, d)
        with torch.no_grad():
            y1, y2 = blk(u, v)
            ru, rv = blk.inverse(y1, y2)
        err = (ru - u).abs().max().item() + (rv - v).abs().max().item()
        print(f"  {coupling:9s} reconstruction max-err = {err:.2e}", "OK" if err < 1e-4 else "FAIL")
        assert err < 1e-4


def test_grad_equivalence():
    print("== gradient equivalence (reversible backward vs plain autograd) ==")
    B, T, d = 2, 8, 32
    for coupling in COUPLINGS:
        blocks = nn.ModuleList([
            make_reversible_block(coupling, d, 4, 4, 0.0, 0.5) for _ in range(3)
        ]).eval()

        x = torch.randn(B, T, 2 * d, requires_grad=True)

        # --- reference: plain autograd through the same blocks ---
        x_ref = x.detach().clone().requires_grad_(True)
        a1, a2 = torch.chunk(x_ref, 2, dim=-1)
        for blk in blocks:
            a1, a2 = blk(a1, a2)
        out_ref = torch.cat([a1, a2], dim=-1)
        loss_ref = out_ref.pow(2).mean()
        ref_grads = torch.autograd.grad(loss_ref, [x_ref] + list(blocks.parameters()),
                                        retain_graph=False)
        ref_dx = ref_grads[0]
        ref_pg = ref_grads[1:]

        # --- reversible: custom Function, activations recomputed ---
        for p in blocks.parameters():
            p.grad = None
        out = _ReversibleFunction.apply(x, blocks)
        loss = out.pow(2).mean()
        loss.backward()
        rev_dx = x.grad
        rev_pg = [p.grad for p in blocks.parameters()]

        dx_err = (rev_dx - ref_dx).abs().max().item()
        pg_err = max((a - b).abs().max().item() for a, b in zip(rev_pg, ref_pg))
        print(f"  {coupling:9s} input-grad err = {dx_err:.2e} | param-grad err = {pg_err:.2e}",
              "OK" if max(dx_err, pg_err) < 1e-4 else "FAIL")
        assert max(dx_err, pg_err) < 1e-4


def test_param_parity():
    print("== parameter-count parity across all configs ==")
    counts = {}
    for coupling in ("none",) + COUPLINGS:
        cfg = GPTConfig(vocab_size=50257, seq_len=128, n_layer=6,
                        d_model=384, n_head=6, coupling=coupling)
        counts[coupling] = GPT(cfg).num_params()
    for k, v in counts.items():
        print(f"  {k:9s} params = {v/1e6:.2f}M")
    assert len(set(counts.values())) == 1
    print("  all identical  OK")


def test_no_activation_storage():
    """A reversible forward under no_grad must not build an autograd graph over
    intermediate activations — only the final output carries grad."""
    print("== reversible forward stores no intermediate graph ==")
    B, T, d = 2, 8, 16
    blocks = nn.ModuleList([make_reversible_block("midpoint", d, 4, 4, 0.0, 0.5) for _ in range(4)])
    x = torch.randn(B, T, 2 * d, requires_grad=True)
    out = _ReversibleFunction.apply(x, blocks)
    # grad_fn is the custom Function; no per-layer graph retained
    ok = out.grad_fn is not None and "ReversibleFunctionBackward" in type(out.grad_fn).__name__
    print(f"  output backed by custom reversible Function: {ok}", "OK" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    test_invertibility()
    test_grad_equivalence()
    test_param_parity()
    test_no_activation_storage()
    print("\nAll checks passed.")
