"""
Train the 20M LM. One script, three configs via --coupling {none,euler,midpoint}.

Logs the four metrics the assignment asks for and appends one JSON line to
results.jsonl per run:
    coupling, batch_size, seq_len, params, final_train_loss, val_loss,
    tokens_per_sec, peak_mem_gb, max_batch (if using the sweep).

Keep seq_len, max_iters and the data identical across the three runs so the
comparison is clean — only change --coupling and --batch_size.

Examples
--------
  python train.py --coupling none     --batch_size 32 --max_iters 3000
  python train.py --coupling euler    --batch_size 32 --max_iters 3000
  python train.py --coupling midpoint --batch_size 32 --max_iters 3000
"""
import os
import json
import time
import math
import argparse
import numpy as np
import torch
from model import GPT, GPTConfig


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, batch_size, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, block_size, batch_size, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def build_model(args, vocab_size, device):
    cfg = GPTConfig(vocab_size=vocab_size, seq_len=args.seq_len,
                    n_layer=args.n_layer, d_model=args.d_model,
                    n_head=args.n_head, mlp_ratio=4, dropout=args.dropout,
                    coupling=args.coupling, h=args.h)
    model = GPT(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model, cfg


def run(args):
    device = args.device
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    meta_path = os.path.join(args.data_dir, "meta.json")
    vocab_size = 50257
    if os.path.exists(meta_path):
        vocab_size = json.load(open(meta_path))["vocab_size"]
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    model, cfg = build_model(args, vocab_size, device)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = raw.num_params()
    print(f"coupling={args.coupling} | params={n_params/1e6:.2f}M | "
          f"batch={args.batch_size} seq={args.seq_len} device={device}")

    opt = raw.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95),
                                  "cuda" if device.startswith("cuda") else "cpu")
    use_amp = device.startswith("cuda")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t0 = time.time()
    tokens_seen = 0
    last_loss = float("nan")
    for it in range(args.max_iters):
        lr = lr_at(it, args.warmup, args.max_iters, args.lr, args.lr / 10)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.seq_len, args.batch_size, device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tokens_seen += args.batch_size * args.seq_len
        last_loss = loss.item()
        if it % args.log_every == 0 or it == args.max_iters - 1:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-9)
            print(f"  it {it:5d} | loss {last_loss:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens_per_sec = tokens_seen / max(elapsed, 1e-9)
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else 0.0
    val_loss = estimate_val_loss(model, val_data, args.seq_len, args.batch_size, device) \
        if args.eval_val else None

    result = {
        "coupling": args.coupling,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_layer": args.n_layer,
        "d_model": args.d_model,
        "params_millions": round(n_params / 1e6, 3),
        "max_iters": args.max_iters,
        "final_train_loss": round(last_loss, 4),
        "val_loss": round(val_loss, 4) if val_loss is not None else None,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_mem_gb": round(peak_mem_gb, 3),
    }
    print("RESULT " + json.dumps(result))
    with open(args.results, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling", choices=["none", "euler", "midpoint", "leapfrog"], default="none")
    ap.add_argument("--h", type=float, default=0.5, help="integrator step size (midpoint uses 2h, leapfrog h^2)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_val", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
