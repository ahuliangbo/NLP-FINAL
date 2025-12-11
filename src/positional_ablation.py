import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from models.decoder_only import DecoderConfig, MiniDecoder
from utils import SimpleTokenizer, set_seed

sns.set_style("whitegrid")


class SimpleDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: SimpleTokenizer, max_length: int = 128):
        self.sequences = [s for s in sequences if s.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.sequences[idx], add_special_tokens=True)
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return tokens


def collate_batch(batch, tokenizer: SimpleTokenizer):
    padded, attention = tokenizer.pad(batch)
    return {
        "input_ids": padded,
        "attention_mask": attention,
        "labels": padded.clone()
    }


def load_sequences(path: Path, max_lines: int = None) -> List[str]:
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            s = line.strip()
            if s:
                sequences.append(s)
    return sequences


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += outputs["accuracy"].item()
        steps += 1
    
    return total_loss / max(1, steps), total_acc / max(1, steps)


def train_model(
    sequences: List[str],
    pos_type: str,
    epochs: int = 5,
    batch_size: int = 32,
    embed_dim: int = 128,
    num_layers: int = 4,
    max_seq_len: int = 128,
    lr: float = 3e-4,
    device: torch.device = None
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Training: Positional Encoding = {pos_type.upper()}")
    print(f"{'='*70}")
    
    tokenizer = SimpleTokenizer(sequences)
    dataset = SimpleDataset(sequences, tokenizer, max_length=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer)
    )
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Positional encoding: {pos_type}")
    print(f"Embed dim: {embed_dim}, Layers: {num_layers}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    history = {
        'losses': [], 'accuracies': [], 'epoch_times': [], 'total_time': 0,
        'model': model, 'tokenizer': tokenizer
    }
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        avg_loss, avg_acc = train_epoch(model, loader, optimizer, device)
        epoch_time = time.time() - epoch_start
        history['losses'].append(avg_loss)
        history['accuracies'].append(avg_acc)
        history['epoch_times'].append(epoch_time)
        history['total_time'] += epoch_time
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Time: {epoch_time:.2f}s")
    
    print(f"Training completed in {history['total_time']:.2f}s")
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Best loss: {min(history['losses']):.4f}")
    
    return history


@torch.no_grad()
def evaluate_perplexity(
    model: MiniDecoder,
    tokenizer: SimpleTokenizer,
    sequences: List[str],
    length: int,
    device: torch.device,
    batch_size: int = 32
) -> float:
    model.eval()
    pad_id = tokenizer.token_id(tokenizer.pad_token)
    eos_id = tokenizer.token_id(tokenizer.eos_token)
    
    inputs = []
    for s in sequences:
        toks = tokenizer.encode(s, add_special_tokens=True)
        if len(toks) >= length:
            t = toks[:length]
            if length > 0:
                t[-1] = eos_id
        else:
            t = toks + [pad_id] * (length - len(toks))
        if any(tok_id >= tokenizer.vocab_size or tok_id < 0 for tok_id in t):
            continue
        inputs.append(t)
    
    if not inputs:
        return float('inf')
    
    total_loss = 0.0
    total_tokens = 0
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        padded, attention = tokenizer.pad(batch)
        input_ids = padded.to(device)
        attention_mask = attention.to(device)
        labels = padded.to(device)
        out = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"].item()
        num_tokens = attention_mask.sum().item()
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return ppl


def plot_training_curves(results: Dict[str, Dict], output_dir: Path):
    print(f"\n{'='*70}")
    print("Generating Training Curves")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for pos_type, history in results.items():
        axes[0].plot(history['losses'], marker='o', label=pos_type, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss by Positional Encoding', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for pos_type, history in results.items():
        axes[1].plot(history['accuracies'], marker='o', label=pos_type, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy by Positional Encoding', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'pos_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_length_generalization(perplexities: Dict[str, List[float]], lengths: List[int], output_dir: Path):
    print(f"\n{'='*70}")
    print("Generating Length Generalization Plot")
    print(f"{'='*70}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for pos_type, ppls in perplexities.items():
        ax.plot(lengths, ppls, marker='o', label=pos_type, linewidth=2, markersize=8)
    
    ax.set_xlabel('Sequence Length', fontsize=13)
    ax.set_ylabel('Perplexity', fontsize=13)
    ax.set_title('Length Generalization: Perplexity vs Sequence Length', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    train_length = 128
    ax.axvline(x=train_length, color='red', linestyle='--', alpha=0.5, label=f'Training Length ({train_length})')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = output_dir / 'pos_length_generalization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved length generalization plot to {save_path}")


def plot_comparison_summary(results: Dict[str, Dict], perplexities: Dict[str, List[float]], output_dir: Path):
    print(f"\n{'='*70}")
    print("Generating Comparison Summary")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pos_types = list(results.keys())
    colors = sns.color_palette("Set2", len(pos_types))
    
    final_losses = [results[p]['losses'][-1] for p in pos_types]
    axes[0, 0].bar(pos_types, final_losses, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Positional Encoding', fontsize=11)
    axes[0, 0].set_ylabel('Final Training Loss', fontsize=11)
    axes[0, 0].set_title('Final Training Loss Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    total_times = [results[p]['total_time'] for p in pos_types]
    axes[0, 1].bar(pos_types, total_times, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Positional Encoding', fontsize=11)
    axes[0, 1].set_ylabel('Total Training Time (s)', fontsize=11)
    axes[0, 1].set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    convergence = {}
    for pos_type, history in results.items():
        target = history['accuracies'][-1] * 0.95
        for i, acc in enumerate(history['accuracies']):
            if acc >= target:
                convergence[pos_type] = i + 1
                break
        else:
            convergence[pos_type] = len(history['accuracies'])
    
    axes[1, 0].bar(list(convergence.keys()), list(convergence.values()),
                   color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Positional Encoding', fontsize=11)
    axes[1, 0].set_ylabel('Epochs to 95% Final Accuracy', fontsize=11)
    axes[1, 0].set_title('Convergence Speed Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    avg_ppls = {pos: np.mean(ppls) for pos, ppls in perplexities.items()}
    axes[1, 1].bar(list(avg_ppls.keys()), list(avg_ppls.values()),
                   color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Positional Encoding', fontsize=11)
    axes[1, 1].set_ylabel('Average Perplexity', fontsize=11)
    axes[1, 1].set_title('Average Perplexity Across Lengths', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / 'pos_comparison_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison summary to {save_path}")


def print_results_table(results: Dict[str, Dict], perplexities: Dict[str, List[float]]):
    print(f"\n{'='*70}")
    print("Positional Encoding Ablation Results")
    print(f"{'='*70}\n")
    
    data = []
    for pos_type, history in results.items():
        avg_ppl = np.mean(perplexities[pos_type])
        data.append({
            "Pos. Encoding": pos_type,
            "Final Loss": f"{history['losses'][-1]:.4f}",
            "Best Loss": f"{min(history['losses']):.4f}",
            "Final Acc": f"{history['accuracies'][-1]:.4f}",
            "Train Time (s)": f"{history['total_time']:.1f}",
            "Avg PPL": f"{avg_ppl:.2f}",
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()


def print_perplexity_table(perplexities: Dict[str, List[float]], lengths: List[int]):
    print(f"\n{'='*70}")
    print("Perplexity by Sequence Length")
    print(f"{'='*70}\n")
    
    data = []
    for pos_type, ppls in perplexities.items():
        row = {"Pos. Encoding": pos_type}
        for length, ppl in zip(lengths, ppls):
            row[f"L={length}"] = f"{ppl:.2f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()


def run_ablation_study(data_path: Path, output_dir: Path, max_lines: int = 3000):
    print(f"\n{'#'*70}")
    print("POSITIONAL ENCODING ABLATION STUDY")
    print(f"{'#'*70}")
    
    set_seed(42)
    sequences = load_sequences(data_path, max_lines=max_lines)
    random.shuffle(sequences)
    
    split1 = int(0.9 * len(sequences))
    split2 = int(0.95 * len(sequences))
    
    train_seqs = sequences[:split1]
    val_seqs = sequences[split1:split2]
    test_seqs = sequences[split2:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Train samples: {len(train_seqs)}")
    print(f"Val samples: {len(val_seqs)}")
    print(f"Test samples: {len(test_seqs)}")
    
    pos_types = ["sinusoidal", "learned", "rotary"]
    results = {}
    
    for pos_type in pos_types:
        results[pos_type] = train_model(
            train_seqs,
            pos_type=pos_type,
            epochs=5,
            batch_size=32,
            embed_dim=128,
            num_layers=4,
            max_seq_len=128,
            device=device
        )
    
    lengths = [64, 128, 192, 256]
    perplexities = {p: [] for p in pos_types}
    
    for pos_type in pos_types:
        print(f"\nEvaluating {pos_type}:")
        model = results[pos_type]['model']
        tokenizer = results[pos_type]['tokenizer']
        for length in lengths:
            ppl = evaluate_perplexity(model, tokenizer, val_seqs[:200], length, device)
            perplexities[pos_type].append(ppl)
            print(f"  Length {length}: PPL = {ppl:.2f}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(results, output_dir)
    plot_length_generalization(perplexities, lengths, output_dir)
    plot_comparison_summary(results, perplexities, output_dir)
    
    print_results_table(results, perplexities)
    print_perplexity_table(perplexities, lengths)
    
    best_loss = min(results.items(), key=lambda x: min(x[1]['losses']))
    best_ppl = min(perplexities.items(), key=lambda x: np.mean(x[1]))
    fastest = min(results.items(), key=lambda x: x[1]['total_time'])
    
    print(f"Best model (lowest training loss): {best_loss[0]}")
    print(f"Best model (lowest avg perplexity): {best_ppl[0]}")
    print(f"Fastest model: {fastest[0]}")
    
    print(f"\n{'='*70}")
    print("Ablation study completed!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*70}")
    
    return results, perplexities


if __name__ == "__main__":
    data_path = Path("data/wikitext2_5k.txt")
    output_dir = Path("outputs")
    run_ablation_study(data_path, output_dir, max_lines=3000)
