"""
Quick Colab-friendly training script for models/qwenold.py.

Goal: get a tiny decoder-only Transformer training run that finishes in ~5 minutes.
Dataset: Tiny Shakespeare (plain text) from Karpathy's char-rnn repo.

Usage (Colab):
  1) Put this repo in Colab (upload zip or git clone).
  2) Run: !python train_qwenold_colab.py --train-seconds 300
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from models.qwenold import QwenModel


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _download_if_needed(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(url, path)


@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def _build_vocab(text: str) -> CharVocab:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return CharVocab(stoi=stoi, itos=itos)


def _make_splits(text: str, *, train_frac: float = 0.9) -> Tuple[str, str]:
    n = int(len(text) * train_frac)
    return text[:n], text[n:]


def _get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sample random contiguous sequences.
    ix = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    iters: int = 20,
) -> Tuple[float, float]:
    model.eval()
    out = []
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(iters):
            xb, yb = _get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    model.train()
    return out[0], out[1]


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    vocab: CharVocab,
    *,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
) -> str:
    model.eval()
    ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)[:, -1, :]  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return vocab.decode(ids[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seconds", type=float, default=300.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Keep defaults small so it runs on a free Colab GPU in a few minutes.
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_path = os.path.join("data", "tinyshakespeare.txt")
    _download_if_needed(TINY_SHAKESPEARE_URL, data_path)
    text = open(data_path, "r", encoding="utf-8").read()

    # Trim text a bit to reduce tokenization overhead and keep runtime consistent.
    # (Tiny Shakespeare is ~1.1MB; using 300k chars is enough for a quick demo.)
    text = text[:300_000]

    vocab = _build_vocab(text)
    train_text, val_text = _make_splits(text)
    train_ids = torch.tensor(vocab.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(vocab.encode(val_text), dtype=torch.long)

    model = QwenModel(
        vocab_size=vocab.size,
        hidden_size=args.hidden_size,
        max_seq_len=args.block_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"vocab_size={vocab.size} device={device} amp={use_amp}")
    print(
        f"block={args.block_size} batch={args.batch_size} "
        f"hidden={args.hidden_size} heads={args.num_heads} layers={args.num_layers}"
    )

    start = time.time()
    step = 0
    while True:
        if time.time() - start >= args.train_seconds:
            break

        xb, yb = _get_batch(
            train_ids, batch_size=args.batch_size, block_size=args.block_size, device=device
        )

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            elapsed = time.time() - start
            print(f"step={step:5d} loss={loss.item():.4f} elapsed={elapsed:.1f}s")

        if step > 0 and step % args.eval_every == 0:
            tr, va = _estimate_loss(
                model,
                train_ids,
                val_ids,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
                iters=args.eval_iters,
            )
            print(f"eval: train={tr:.4f} val={va:.4f}")

        step += 1

    print(f"Training done. steps={step} seconds={time.time() - start:.1f}")
    print()
    print(_generate(model, vocab, device=device, prompt="ROMEO:\n", max_new_tokens=300, block_size=args.block_size))


if __name__ == "__main__":
    main()

