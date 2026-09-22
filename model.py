"""
Reversible language model — baseline + three reversible integrators.

Method follows Karami et al., "Reversing Large Language Models for Efficient
Training and Fine-Tuning" (arXiv:2512.02056): reversible LLM layers cast as
time-reversible / symplectic ODE integrators, so hidden states are reconstructed
during backprop instead of stored.

Configs (select with `coupling`):
  "none"     : standard pre-norm Transformer (stores activations)   -> baseline
  "euler"    : Hamiltonian coupled-stream additive coupling (1st order)
                 q = q_prev + Attn(LN1(p_prev))
                 p = p_prev + MLP (LN2(q))
  "midpoint" : explicit-midpoint / leapfrog multi-step integrator
                 p^{l+1} = p^{l-1} + step * f(p^{l})
  "leapfrog" : Stormer-Verlet 2nd-order integrator
                 p^{l+1} = 2 p^{l} - p^{l-1} + step * f(p^{l})

All reversible blocks carry TWO full-width streams (u, v) and reuse the same
Attn + MLP as the baseline, so every config has the SAME parameter count.

NOTE: exact reconstruction requires the layer function to be deterministic, so
reversible modes run with dropout = 0 (enforced below).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Sublayers
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, mlp_ratio * d_model)
        self.proj = nn.Linear(mlp_ratio * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SubLayer(nn.Module):
    """LayerNorm -> fn (pre-norm)."""
    def __init__(self, d_model, fn):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.ln(x))


class LayerFn(nn.Module):
    """Full layer step f(p) = Attn(LN1(p)) + MLP(LN2(p + Attn(...)));
    the residual 'delta' of a standard pre-norm block. Used by the single-stream
    integrators (midpoint, leapfrog). Same params as a baseline block."""
    def __init__(self, d_model, n_head, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, p):
        a = self.attn(self.ln1(p))
        m = self.mlp(self.ln2(p + a))
        return a + m


# ----------------------------------------------------------------------------
# Baseline (non-reversible) block
# ----------------------------------------------------------------------------
class StandardBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio, dropout):
        super().__init__()
        self.attn = SubLayer(d_model, CausalSelfAttention(d_model, n_head, dropout))
        self.ffn = SubLayer(d_model, MLP(d_model, mlp_ratio, dropout))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


# ----------------------------------------------------------------------------
# Reversible blocks: each defines forward(u,v) and exact inverse(u',v').
# The memory-saving backward is shared in the base class.
# ----------------------------------------------------------------------------
class ReversibleBase(nn.Module):
    def backward_pass(self, u2, v2, du2, dv2):
        """Reconstruct inputs from outputs, recompute forward with grad,
        backprop for input grads + accumulate param grads."""
        with torch.no_grad():
            u1, v1 = self.inverse(u2, v2)
        with torch.enable_grad():
            u1 = u1.detach().requires_grad_(True)
            v1 = v1.detach().requires_grad_(True)
            uo, vo = self.forward(u1, v1)
            torch.autograd.backward((uo, vo), (du2, dv2))
        with torch.no_grad():
            du1, dv1 = u1.grad, v1.grad
            u1, v1 = u1.detach(), v1.detach()
        return u1, v1, du1, dv1


class HamiltonianBlock(ReversibleBase):
    """euler: q = q_prev + F(p_prev);  p = p_prev + G(q).
    Here u=q (momentum), v=p (position). F=Attn, G=MLP."""
    def __init__(self, d_model, n_head, mlp_ratio, dropout):
        super().__init__()
        self.F = SubLayer(d_model, CausalSelfAttention(d_model, n_head, dropout))
        self.G = SubLayer(d_model, MLP(d_model, mlp_ratio, dropout))

    def forward(self, u, v):          # (q_prev, p_prev)
        q = u + self.F(v)
        p = v + self.G(q)
        return q, p

    def inverse(self, u2, v2):        # (q, p)
        p_prev = v2 - self.G(u2)
        q_prev = u2 - self.F(p_prev)
        return q_prev, p_prev


class MidpointBlock(ReversibleBase):
    """Explicit midpoint (paper eq.):  p^{l+1} = p^{l-1} + 2h * f(p^{l}).
    State (u,v) = (p_prev, p_cur). The block is evaluated at the INTERMEDIATE
    state p_cur, which the backward pass already holds — that is what makes the
    reverse step  p^{l-1} = p^{l+1} - 2h * f(p^{l})  exact without inverting f.
    `h` is the fixed step size; stable only in a narrow range (see paper)."""
    def __init__(self, d_model, n_head, mlp_ratio, dropout, h=0.5):
        super().__init__()
        self.f = LayerFn(d_model, n_head, mlp_ratio, dropout)
        self.c = 2.0 * h                          # coefficient 2h

    def forward(self, u, v):          # (p_prev, p_cur) -> (p_cur, p_next)
        p_next = u + self.c * self.f(v)
        return v, p_next

    def inverse(self, u2, v2):        # (p_cur, p_next) -> (p_prev, p_cur)
        p_prev = v2 - self.c * self.f(u2)
        return p_prev, u2


class LeapfrogBlock(ReversibleBase):
    """Stormer-Verlet 2nd-order:  p^{l+1} = 2 p^{l} - p^{l-1} + h^2 * f(p^{l}).
    State (u,v) = (p_prev, p_cur)."""
    def __init__(self, d_model, n_head, mlp_ratio, dropout, h=0.5):
        super().__init__()
        self.f = LayerFn(d_model, n_head, mlp_ratio, dropout)
        self.c = h * h                            # coefficient h^2

    def forward(self, u, v):          # (p_prev, p_cur) -> (p_cur, p_next)
        p_next = 2 * v - u + self.c * self.f(v)
        return v, p_next

    def inverse(self, u2, v2):        # (p_cur, p_next) -> (p_prev, p_cur)
        p_prev = 2 * u2 - v2 + self.c * self.f(u2)
        return p_prev, u2


def make_reversible_block(coupling, d_model, n_head, mlp_ratio, dropout, h):
    if coupling == "euler":
        return HamiltonianBlock(d_model, n_head, mlp_ratio, dropout)
    if coupling == "midpoint":
        return MidpointBlock(d_model, n_head, mlp_ratio, dropout, h)
    if coupling == "leapfrog":
        return LeapfrogBlock(d_model, n_head, mlp_ratio, dropout, h)
    raise ValueError(coupling)


# ----------------------------------------------------------------------------
# Memory-saving reversible stack
# ----------------------------------------------------------------------------
class _ReversibleFunction(torch.autograd.Function):
    """Runs a stack of reversible blocks storing ONLY the final output;
    activations are reconstructed block-by-block in backward."""
    @staticmethod
    def forward(ctx, x, blocks):
        ctx.blocks = blocks
        u, v = torch.chunk(x, 2, dim=-1)
        with torch.no_grad():
            for blk in blocks:
                u, v = blk(u, v)
        out = torch.cat([u, v], dim=-1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, = ctx.saved_tensors
        u, v = torch.chunk(out, 2, dim=-1)
        du, dv = torch.chunk(grad_out, 2, dim=-1)
        for blk in reversed(ctx.blocks):
            u, v, du, dv = blk.backward_pass(u, v, du, dv)
        return torch.cat([du, dv], dim=-1), None


class ReversibleStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # seed the two coupled streams from the embedding
        x = torch.cat([x, x], dim=-1)
        x = _ReversibleFunction.apply(x, self.blocks)
        u, v = torch.chunk(x, 2, dim=-1)
        return v  # the "position" stream


# ----------------------------------------------------------------------------
# Full GPT
# ----------------------------------------------------------------------------
class GPTConfig:
    def __init__(self, vocab_size=50257, seq_len=256, n_layer=10, d_model=256,
                 n_head=8, mlp_ratio=4, dropout=0.0, coupling="none", h=0.5):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.coupling = coupling            # none | euler | midpoint | leapfrog
        self.h = h                          # integrator step size (midpoint uses 2h, leapfrog h^2)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        drop = cfg.dropout
        if cfg.coupling != "none" and drop != 0.0:
            print("[model] reversible mode requires deterministic layers; forcing dropout=0")
            drop = 0.0

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(drop)

        if cfg.coupling == "none":
            self.stack = nn.ModuleList([
                StandardBlock(cfg.d_model, cfg.n_head, cfg.mlp_ratio, drop)
                for _ in range(cfg.n_layer)
            ])
            self.reversible = False
        else:
            self.stack = ReversibleStack([
                make_reversible_block(cfg.coupling, cfg.d_model, cfg.n_head,
                                      cfg.mlp_ratio, drop, cfg.h)
                for _ in range(cfg.n_layer)
            ])
            self.reversible = True

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        if self.reversible:
            x = self.stack(x)
        else:
            for blk in self.stack:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas,
                                 fused=(device_type == "cuda"))
