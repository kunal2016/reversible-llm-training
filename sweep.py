"""
Find the maximum stable batch size for a coupling, then report throughput/memory
at that batch. Doubles the batch until CUDA OOM, then does one linear back-off
step. Runs a short burn-in (few iters) at each candidate — fast.

Usage:
  python sweep.py --coupling none
  python sweep.py --coupling euler
  python sweep.py --coupling midpoint

Appends the winning config to results.jsonl (tagged with "max_batch": true) and
prints a one-line summary you can paste straight into the report.
"""
import json
import argparse
import torch
from train import build_argparser, run


def try_batch(base_args, batch):
    args = argparse.Namespace(**vars(base_args))
    args.batch_size = batch
    args.max_iters = base_args.probe_iters
    args.eval_val = False
    args.results = "/dev/null"  # don't log probe runs
    try:
        res = run(args)
        torch.cuda.empty_cache()
        return res
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise


def main():
    ap = build_argparser()
    ap.add_argument("--start_batch", type=int, default=8)
    ap.add_argument("--probe_iters", type=int, default=60)
    args = ap.parse_args()

    if not args.device.startswith("cuda"):
        print("WARNING: OOM sweep is only meaningful on CUDA. Running anyway for smoke test.")

    best = None
    batch = args.start_batch
    last_ok = None
    print(f"=== batch-size sweep: coupling={args.coupling} ===")
    while True:
        print(f"-- trying batch_size={batch}")
        res = try_batch(args, batch)
        if res is None:
            print(f"   OOM at batch_size={batch}")
            break
        last_ok = res
        best = batch
        batch *= 2
        if batch > 100000:
            break

    if best is None:
        print("Could not fit even the starting batch size; lower --start_batch.")
        return

    # linear back-off: try to squeeze between best and 2*best
    hi = best * 2
    probe = best + (hi - best) // 2
    while probe > best:
        print(f"-- back-off probe batch_size={probe}")
        res = try_batch(args, probe)
        if res is not None:
            best = probe
            last_ok = res
            probe = best + (hi - best) // 2
        else:
            hi = probe
            probe = best + (hi - best) // 2

    summary = {
        "coupling": args.coupling,
        "max_batch": best,
        "seq_len": args.seq_len,
        "params_millions": last_ok["params_millions"],
        "tokens_per_sec_at_max": last_ok["tokens_per_sec"],
        "peak_mem_gb_at_max": last_ok["peak_mem_gb"],
    }
    print("MAXBATCH " + json.dumps(summary))
    with open(args.results, "a") as f:
        f.write(json.dumps({**summary, "is_max_batch_sweep": True}) + "\n")
    print(f"\n>> {args.coupling}: max stable batch = {best}, "
          f"{last_ok['tokens_per_sec']:,.0f} tok/s, {last_ok['peak_mem_gb']:.2f} GB peak")


if __name__ == "__main__":
    main()
