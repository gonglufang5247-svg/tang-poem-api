"""You should not need to change this file. Run: python evaluate.py

Loads your saved model and reports its loss on val.txt (also provides load_model/measure used by train.py and generate.py).
"""

import argparse
import json
import os

import torch

import student
from model import GPT


def measure(model, val_ids, batch=256):
    T = model.block_size
    device = next(model.parameters()).device
    n_chunks = (len(val_ids) - 1) // (T + 1)
    if n_chunks == 0:
        raise SystemExit("val.txt is shorter than one block — cannot measure loss")
    d = val_ids[: n_chunks * (T + 1)].view(n_chunks, T + 1)
    x, y = d[:, :-1], d[:, 1:]
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, n_chunks, batch):
            xb = x[i : i + batch].to(device)
            yb = y[i : i + batch].to(device)
            total += student.loss(model(xb), yb).item() * len(xb)
    if was_training:
        model.train()
    return total / n_chunks


def load_model(data_dir, out_dir):
    with open(os.path.join(data_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sid = str(cfg.get("student_id", "")).strip()
    with open(os.path.join(out_dir, f"{sid}.json"), encoding="utf-8") as f:
        saved = json.load(f)
    model = GPT(len(saved["chars"]), saved)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(os.path.join(out_dir, f"{sid}.safetensors")))
    return model, saved


def encode(text, chars):
    stoi = {c: i for i, c in enumerate(chars)}
    missing = sorted(set(text) - set(stoi))
    if missing:
        raise SystemExit(f"characters in val.txt not in the vocabulary: {''.join(missing)!r}")
    return torch.tensor([stoi[c] for c in text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="folder holding config.json and val.txt")
    ap.add_argument("--out", default="output", help="folder holding <ID>.safetensors and <ID>.json")
    args = ap.parse_args()

    model, saved = load_model(args.data, args.out)
    val_text = open(os.path.join(args.data, "val.txt"), encoding="utf-8").read()
    val_ids = encode(val_text, saved["chars"])
    print(f"val loss {measure(model, val_ids):.3f}")


if __name__ == "__main__":
    main()
