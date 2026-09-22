"""
Prepare ~50M tokens of training data, nanoGPT-style.

Tokenizes text with the GPT-2 BPE (tiktoken) and writes uint16 binaries:
  <out_dir>/train.bin   (~95%)
  <out_dir>/val.bin     (~5%)
  <out_dir>/meta.json   (vocab_size, token counts)

Two sources:
  --source hf     a HuggingFace dataset streamed until ~N tokens (default)
  --source file   a local .txt file you point at with --path

Examples
--------
  # default: TinyStories, ~50M tokens
  python data.py --tokens 50_000_000

  # wikitext-103 raw
  python data.py --dataset wikitext --config wikitext-103-raw-v1 --tokens 50_000_000

  # your own text file
  python data.py --source file --path corpus.txt --tokens 50_000_000
"""
import os
import json
import argparse
import numpy as np


def get_encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def iter_texts_hf(dataset, config, split, text_field):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    for ex in ds:
        t = ex.get(text_field)
        if t:
            yield t


def iter_texts_file(path):
    # stream a big text file in ~1MB chunks split on blank lines
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if sum(len(b) for b in buf) > 1_000_000:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "file"], default="hf")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--path", default=None, help="local .txt (for --source file)")
    ap.add_argument("--tokens", type=int, default=50_000_000)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = get_encoder()
    eot = enc.eot_token

    if args.source == "file":
        assert args.path, "--path required for --source file"
        texts = iter_texts_file(args.path)
    else:
        texts = iter_texts_hf(args.dataset, args.config, args.split, args.text_field)

    print(f"Tokenizing up to {args.tokens:,} tokens with GPT-2 BPE ...")
    ids = np.empty(args.tokens + 4096, dtype=np.uint16)
    n = 0
    for i, text in enumerate(texts):
        toks = enc.encode_ordinary(text)
        toks.append(eot)
        k = min(len(toks), len(ids) - n)
        ids[n:n + k] = np.array(toks[:k], dtype=np.uint16)
        n += k
        if i % 2000 == 0:
            print(f"  {n:,} tokens", end="\r")
        if n >= args.tokens:
            break
    ids = ids[:n]
    print(f"\nCollected {n:,} tokens.")

    n_val = int(n * args.val_frac)
    val = ids[:n_val]
    train = ids[n_val:]
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    val.tofile(os.path.join(args.out_dir, "val.bin"))

    meta = {"vocab_size": enc.n_vocab, "encoding": "gpt2",
            "train_tokens": int(train.size), "val_tokens": int(val.size)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote train.bin ({train.size:,}), val.bin ({val.size:,}) to {args.out_dir}/")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
