"""Run positional encoding ablation (sinusoidal vs learned vs rotary).

Saves plots: `pos_ablation_loss.png` and `pos_ablation_perplexity.png`.
"""
import math
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.decoder_only import DecoderConfig, MiniDecoder
from utils import SimpleTokenizer, set_seed, pad_sequences


class SimpleDataset:
    def __init__(self, sequences: List[str], tokenizer: SimpleTokenizer, max_length: int = None):
        self.sequences = [s for s in sequences if s.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.sequences[idx], add_special_tokens=True)
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        return tokens


def collate_batch(batch, tokenizer: SimpleTokenizer):
    padded, attention = tokenizer.pad(batch)
    return {"input_ids": padded, "attention_mask": attention, "labels": padded.clone()}


def load_lines(path: Path, max_lines: int = None) -> List[str]:
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            s = line.strip()
            if s:
                seqs.append(s)
    return seqs


def train_model(sequences, data_path, pos_type, epochs=5, batch_size=16, embed_dim=128, num_layers=4, max_seq_len=128, lr=3e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(sequences)
    config = DecoderConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=max(1, embed_dim // 32),
        ff_dim=embed_dim * 2,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=0.1,
        pad_token_id=tokenizer.token_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_id(tokenizer.eos_token),
        pos_type=pos_type,
    )
    model = MiniDecoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = SimpleDataset(sequences, tokenizer, max_length=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer))

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        avg = total_loss / max(1, steps)
        print(f"pos={pos_type} epoch={epoch} loss={avg:.4f}")
        losses.append(avg)
    return model, tokenizer, losses


@torch.no_grad()
def eval_perplexity(model: MiniDecoder, tokenizer: SimpleTokenizer, sequences: List[str], length: int, device=None, batch_size=32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs = []
    for s in sequences:
        toks = tokenizer.encode(s, add_special_tokens=True)
        # make sequence of requested length by repeating/ truncating
        if len(toks) >= length:
            t = toks[:length]
        else:
            # repeat tokens (excluding special tokens) to reach length
            body = toks[1:-1] if len(toks) > 2 else toks
            if not body:
                body = [tokenizer.token_id(tokenizer.mask_token)]
            rep = []
            while len(rep) + 2 < length:
                rep.extend(body)
            t = [toks[0]] + rep[: max(0, length - 2)] + [toks[-1]]
            t = t[:length]
        inputs.append(t)
    # batch eval
    total_loss = 0.0
    total_tokens = 0
    device = model.lm_head.weight.device
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        padded, attention = tokenizer.pad(batch)
        input_ids = padded.to(device)
        attention_mask = attention.to(device)
        labels = padded.to(device)
        out = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(out["loss"].item())
        # loss returned is average per token in batch, multiply
        # approximate total tokens = batch_size * length
        bs = input_ids.size(0)
        total_loss += loss * bs
        total_tokens += bs
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return ppl


def run_ablation(data_path: Path, output_dir: Path, max_lines=2000):
    set_seed(42)
    seqs = load_lines(data_path, max_lines=max_lines)
    random.shuffle(seqs)
    train_seqs = seqs[: int(0.9 * len(seqs))]
    val_seqs = seqs[int(0.9 * len(seqs)) : int(0.95 * len(seqs))]
    test_seqs = seqs[int(0.95 * len(seqs)) :]

    pos_types = ["sinusoidal", "learned", "rotary"]
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pos in pos_types:
        print(f"\n=== Training pos_type={pos} ===")
        model, tokenizer, losses = train_model(train_seqs, data_path, pos_type=pos, epochs=5, batch_size=32, embed_dim=128, num_layers=4, max_seq_len=128, device=device)
        results[pos] = {"model": model, "tokenizer": tokenizer, "losses": losses}

    # evaluate perp on lengths
    lengths = [64, 128, 192, 256]
    perps = {p: [] for p in pos_types}
    for pos in pos_types:
        model = results[pos]["model"]
        tokenizer = results[pos]["tokenizer"]
        for L in lengths:
            ppl = eval_perplexity(model, tokenizer, val_seqs[:200], length=L, device=device)
            perps[pos].append(ppl)
            print(f"pos={pos} len={L} ppl={ppl:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # plot losses
    plt.figure(figsize=(8, 4))
    for pos in pos_types:
        plt.plot(results[pos]["losses"], marker="o", label=pos)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Positional Encoding Ablation: Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "pos_ablation_loss.png", dpi=150)
    print(f"Saved loss plot to {output_dir / 'pos_ablation_loss.png'}")

    # plot perplexity vs length
    plt.figure(figsize=(8, 4))
    for pos in pos_types:
        plt.plot(lengths, perps[pos], marker="o", label=pos)
    plt.xlabel("Sequence length")
    plt.ylabel("Perplexity")
    plt.title("Positional Encoding Ablation: Validation Perplexity by Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "pos_ablation_perplexity.png", dpi=150)
    print(f"Saved perplexity plot to {output_dir / 'pos_ablation_perplexity.png'}")

    # save results json
    summary = {p: {"losses": results[p]["losses"], "perplexities": perps[p]} for p in pos_types}
    import json

    with open(output_dir / "pos_ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {output_dir / 'pos_ablation_results.json'}")


if __name__ == "__main__":
    run_ablation(Path(__file__).parent / "data" / "wikitext2_5k.txt", Path(__file__).parent / "outputs", max_lines=3000)
