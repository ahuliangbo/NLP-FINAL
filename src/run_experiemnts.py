import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

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

class PatternDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: SimpleTokenizer, max_length: int = 128):
        self.sequences = [seq for seq in sequences if seq.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        tokens = self.tokenizer.encode(self.sequences[idx], add_special_tokens=True)
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return tokens

def collate_batch(batch, tokenizer: SimpleTokenizer):
    padded, attention = tokenizer.pad(batch)
    return {
        "input_ids": padded,
        "attention_mask": attention,
        "labels": padded.clone(),
    }

def load_sequences(path: Path, max_lines: int = None) -> List[str]:
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                sequences.append(stripped)
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

def train_model(config: DecoderConfig, sequences: List[str], optimizer_name: str = "adam",
                epochs: int = 5, batch_size: int = 16, lr: float = 3e-4, device: torch.device = None) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining: {config.embed_dim}D-{config.num_layers}L-{optimizer_name.upper()}")
    tokenizer = SimpleTokenizer(sequences)
    dataset = PatternDataset(sequences, tokenizer, max_length=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer))
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.token_id(tokenizer.pad_token)
    config.bos_token_id = tokenizer.token_id(tokenizer.bos_token)
    config.eos_token_id = tokenizer.token_id(tokenizer.eos_token)
    model = MiniDecoder(config).to(device)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    history = {'losses': [], 'accuracies': [], 'epoch_times': [], 'total_time': 0}
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        avg_loss, avg_acc = train_epoch(model, loader, optimizer, device)
        epoch_time = time.time() - epoch_start
        history['losses'].append(avg_loss)
        history['accuracies'].append(avg_acc)
        history['epoch_times'].append(epoch_time)
        history['total_time'] += epoch_time
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Time: {epoch_time:.2f}s")
    if torch.cuda.is_available():
        history['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
    else:
        history['peak_memory_gb'] = 0
    print(f"Training completed in {history['total_time']:.2f}s | Final loss: {history['losses'][-1]:.4f}")
    return history

def plot_model_size_comparison(results: Dict[str, Dict], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name, history in results.items():
        axes[0, 0].plot(history['losses'], marker='o', label=name)
        axes[0, 1].plot(history['accuracies'], marker='o', label=name)
    axes[1, 0].bar(list(results.keys()), [results[n]['total_time'] for n in results])
    final_losses = [results[n]['losses'][-1] for n in results]
    final_accs = [results[n]['accuracies'][-1] for n in results]
    x = np.arange(len(results))
    width = 0.35
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(x - width/2, final_losses, width, color='coral', alpha=0.7)
    ax2.bar(x + width/2, final_accs, width, color='skyblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'model_size_comparison.png')
    plt.close()
    print(f"Saved model size comparison to {output_dir / 'model_size_comparison.png'}")

def plot_optimizer_comparison(results: Dict[str, Dict], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for opt_name, history in results.items():
        axes[0, 0].plot(history['losses'], marker='o', label=opt_name.upper())
        axes[0, 1].plot(history['accuracies'], marker='o', label=opt_name.upper())
    axes[1, 0].bar([n.upper() for n in results], [results[n]['total_time'] for n in results])
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'optimizer_comparison.png')
    plt.close()
    print(f"Saved optimizer comparison to {output_dir / 'optimizer_comparison.png'}")

def print_results_table(results: Dict[str, Dict], experiment_name: str):
    data = []
    for name, history in results.items():
        data.append({
            "Model/Optimizer": name,
            "Total Time (s)": f"{history['total_time']:.1f}",
            "Avg Epoch (s)": f"{np.mean(history['epoch_times']):.1f}",
            "Final Loss": f"{history['losses'][-1]:.4f}",
            "Best Loss": f"{min(history['losses']):.4f}",
            "Final Acc": f"{history['accuracies'][-1]:.4f}",
            "Peak Mem (GB)": f"{history.get('peak_memory_gb', 0):.2f}",
        })
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

def run_model_size_experiment(data_path: Path, output_dir: Path):
    set_seed(42)
    sequences = load_sequences(data_path, max_lines=5000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    results["small"] = train_model(DecoderConfig(vocab_size=1000, embed_dim=64, num_heads=2, ff_dim=128, num_layers=2, dropout=0.1, max_seq_len=128), sequences, "adam", epochs=5, device=device)
    results["medium"] = train_model(DecoderConfig(vocab_size=1000, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, dropout=0.1, max_seq_len=128), sequences, "adam", epochs=5, device=device)
    results["large"] = train_model(DecoderConfig(vocab_size=1000, embed_dim=256, num_heads=8, ff_dim=512, num_layers=6, dropout=0.1, max_seq_len=128), sequences, "adam", epochs=5, device=device)
    plot_model_size_comparison(results, output_dir)
    print_results_table(results, "Model Size Scaling")
    best_loss = min(results.items(), key=lambda x: min(x[1]['losses']))
    fastest = min(results.items(), key=lambda x: x[1]['total_time'])
    print(f"Best model (lowest loss): {best_loss[0]} | Fastest model: {fastest[0]}")
    return results

def run_optimizer_experiment(data_path: Path, output_dir: Path):
    set_seed(42)
    sequences = load_sequences(data_path, max_lines=5000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_base = DecoderConfig(vocab_size=1000, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, dropout=0.1, max_seq_len=128)
    results = {}
    for opt_type in ["adam", "adamw", "sgd"]:
        results[opt_type] = train_model(config_base, sequences, opt_type, epochs=5, device=device)
    plot_optimizer_comparison(results, output_dir)
    print_results_table(results, "Optimizer Comparison")
    best_opt = min(results.items(), key=lambda x: min(x[1]['losses']))
    fastest_opt = min(results.items(), key=lambda x: x[1]['total_time'])
    print(f"Best optimizer: {best_opt[0].upper()} | Fastest optimizer: {fastest_opt[0].upper()}")
    return results

if __name__ == "__main__":
    data_path = Path("data/wikitext2_5k.txt")
    output_dir = Path("outputs")
    size_results = run_model_size_experiment(data_path, output_dir)
    opt_results = run_optimizer_experiment(data_path, output_dir)
    print(f"\nAll experiments completed! Results saved to {output_dir}")
