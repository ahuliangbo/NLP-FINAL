import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.decoder_only import DecoderConfig, MiniDecoder
from utils import SimpleTokenizer, set_seed

@dataclass
class ExperimentConfig:
    task: str = "mnli"
    max_samples_train: int = 50000
    max_samples_test: int = 5000
    max_length: int = 128
    
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    
    mini_embed_dim: int = 128
    mini_num_layers: int = 6
    mini_num_heads: int = 4
    
    output_dir: Path = Path("outputs/")
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NLIDataset(Dataset):
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }


def load_nli_data(config: ExperimentConfig) -> Tuple[List[str], List[int], List[str], List[int], int]:
    print(f"Loading {config.task.upper()} dataset...")
    
    if config.task == "mnli":
        dataset = load_dataset("glue", "mnli")
        train_split = dataset['train']
        test_split = dataset['validation_matched']
        
        train_texts = [
            f"{premise} [SEP] {hypothesis}" 
            for premise, hypothesis in zip(
                train_split['premise'][:config.max_samples_train],
                train_split['hypothesis'][:config.max_samples_train]
            )
        ]
        train_labels = train_split['label'][:config.max_samples_train]
        
        test_texts = [
            f"{premise} [SEP] {hypothesis}" 
            for premise, hypothesis in zip(
                test_split['premise'][:config.max_samples_test],
                test_split['hypothesis'][:config.max_samples_test]
            )
        ]
        test_labels = test_split['label'][:config.max_samples_test]
        num_labels = 3
        
    elif config.task == "qnli":
        dataset = load_dataset("glue", "qnli")
        train_split = dataset['train']
        test_split = dataset['validation']
        
        train_texts = [
            f"{question} [SEP] {sentence}" 
            for question, sentence in zip(
                train_split['question'][:config.max_samples_train],
                train_split['sentence'][:config.max_samples_train]
            )
        ]
        train_labels = train_split['label'][:config.max_samples_train]
        
        test_texts = [
            f"{question} [SEP] {sentence}" 
            for question, sentence in zip(
                test_split['question'][:config.max_samples_test],
                test_split['sentence'][:config.max_samples_test]
            )
        ]
        test_labels = test_split['label'][:config.max_samples_test]
        num_labels = 2
        
    elif config.task == "qqp":
        dataset = load_dataset("glue", "qqp")
        train_split = dataset['train']
        test_split = dataset['validation']
        
        train_texts = [
            f"{q1} [SEP] {q2}" 
            for q1, q2 in zip(
                train_split['question1'][:config.max_samples_train],
                train_split['question2'][:config.max_samples_train]
            )
        ]
        train_labels = train_split['label'][:config.max_samples_train]
        
        test_texts = [
            f"{q1} [SEP] {q2}" 
            for q1, q2 in zip(
                test_split['question1'][:config.max_samples_test],
                test_split['question2'][:config.max_samples_test]
            )
        ]
        test_labels = test_split['label'][:config.max_samples_test]
        num_labels = 2
    
    else:
        raise ValueError(f"Unknown task: {config.task}")
    
    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    print(f"Number of classes: {num_labels}")
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    print(f"Sample text: {train_texts[0][:100]}...")
    
    return train_texts, train_labels, test_texts, test_labels, num_labels


class MiniDecoderClassifier(nn.Module):
    def __init__(self, config: DecoderConfig, num_classes: int):
        super().__init__()
        self.decoder = MiniDecoder(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.embed_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask, return_attentions=False):
        if return_attentions:
            hidden, attentions = self.decoder.forward(input_ids, attention_mask, return_attentions=True)
        else:
            hidden = self.decoder.forward(input_ids, attention_mask)
            attentions = None
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden.size(0)
        last_hidden = hidden[torch.arange(batch_size), sequence_lengths]
        
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        
        if return_attentions:
            return logits, attentions
        return logits


def prepare_mini_model(train_texts: List[str], config: ExperimentConfig, num_labels: int) -> Tuple[MiniDecoderClassifier, SimpleTokenizer]:
    print("\n--Preparing Mini Model--")
    
    tokenizer = SimpleTokenizer(train_texts)
    print(f"Mini model vocab size: {tokenizer.vocab_size}")
    
    decoder_config = DecoderConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.mini_embed_dim,
        num_heads=config.mini_num_heads,
        ff_dim=config.mini_embed_dim * 4,
        num_layers=config.mini_num_layers,
        max_seq_len=config.max_length,
        dropout=0.1,
        pad_token_id=tokenizer.token_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_id(tokenizer.eos_token),
        pos_type="rotary"
    )
    
    model = MiniDecoderClassifier(decoder_config, num_classes=num_labels)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def prepare_pretrained_model(model_name: str = "distilgpt2", num_labels: int = 3):
    """Load pretrained model from Hugging Face."""
    print(f"\n--Preparing Pretrained Model: {model_name}--")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
        output_attentions=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def train_mini_model(
    model: MiniDecoderClassifier,
    tokenizer: SimpleTokenizer,
    train_texts: List[str],
    train_labels: List[int],
    config: ExperimentConfig
) -> Dict:
    print("\n--Training Mini Model--")
    
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    dataset = NLIDataset(train_texts, train_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    history = {'loss': [], 'accuracy': [], 'time_per_epoch': []}
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            texts = batch['text']
            labels = torch.as_tensor(batch['label'], dtype=torch.long).to(config.device)
            
            token_ids = [tokenizer.encode(text, add_special_tokens=True)[:config.max_length] 
                        for text in texts]
            padded, attention_mask = tokenizer.pad(token_ids, max_length=config.max_length)
            input_ids = padded.to(config.device)
            attention_mask = attention_mask.to(config.device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['time_per_epoch'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Time: {epoch_time:.2f}s")
    
    return history


def train_pretrained_model(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    config: ExperimentConfig
) -> Dict:
    print("\n--Fine-tuning Pretrained Model--")
    
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    dataset = NLIDataset(train_texts, train_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    history = {'loss': [], 'accuracy': [], 'time_per_epoch': []}
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            texts = batch['text']
            labels = torch.as_tensor(batch['label'], dtype=torch.long).to(config.device)
            
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(config.device)
            attention_mask = encoded['attention_mask'].to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['time_per_epoch'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Time: {epoch_time:.2f}s")
    
    return history


@torch.no_grad()
def evaluate_mini_model(
    model: MiniDecoderClassifier,
    tokenizer: SimpleTokenizer,
    test_texts: List[str],
    test_labels: List[int],
    config: ExperimentConfig
) -> Dict:
    print("\n--Evaluating Mini Model--")
    model.eval()
    model = model.to(config.device)
    
    dataset = NLIDataset(test_texts, test_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    total_time = 0
    
    for batch in dataloader:
        start_time = time.time()
        
        texts = batch['text']
        labels = batch['label']
        
        token_ids = [tokenizer.encode(text, add_special_tokens=True)[:config.max_length] 
                    for text in texts]
        padded, attention_mask = tokenizer.pad(token_ids, max_length=config.max_length)
        input_ids = padded.to(config.device)
        attention_mask = attention_mask.to(config.device)
        
        logits = model(input_ids, attention_mask)
        predictions = logits.argmax(dim=1).cpu().tolist()
        
        total_time += time.time() - start_time
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels)
    avg_inference_time = total_time / len(dataloader)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }


@torch.no_grad()
def evaluate_pretrained_model(
    model,
    tokenizer,
    test_texts: List[str],
    test_labels: List[int],
    config: ExperimentConfig
) -> Dict:
    print("\n--Evaluating Pretrained Model--")
    model.eval()
    model = model.to(config.device)
    
    dataset = NLIDataset(test_texts, test_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    total_time = 0
    
    for batch in dataloader:
        start_time = time.time()
        
        texts = batch['text']
        labels = batch['label']
        
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(config.device)
        attention_mask = encoded['attention_mask'].to(config.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = logits.argmax(dim=1).cpu().tolist()
        
        total_time += time.time() - start_time
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels)
    avg_inference_time = total_time / len(dataloader)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }


@torch.no_grad()
def visualize_attention_mini(
    model: MiniDecoderClassifier,
    tokenizer: SimpleTokenizer,
    text: str,
    config: ExperimentConfig,
    save_path: Path,
    layer_idx: int = -1
):
    print(f"\n--Generating Attention Heatmap (Mini Model)--")
    model.eval()
    model = model.to(config.device)
    
    token_ids = tokenizer.encode(text, add_special_tokens=True)[:config.max_length]
    
    if hasattr(tokenizer, 'id_to_token') and callable(tokenizer.id_to_token):
        tokens = [tokenizer.id_to_token(tid) for tid in token_ids]
    elif hasattr(tokenizer, 'id_to_token') and isinstance(tokenizer.id_to_token, dict):
        tokens = [tokenizer.id_to_token.get(tid, f"<unk_{tid}>") for tid in token_ids]
    else:
        tokens = [f"tok_{tid}" for tid in token_ids]
    
    tokens = [tok[:20] if len(tok) > 20 else tok for tok in tokens]
    
    padded, attention_mask = tokenizer.pad([token_ids], max_length=config.max_length)
    input_ids = padded.to(config.device)
    attention_mask = attention_mask.to(config.device)
    
    try:
        logits, attentions = model(input_ids, attention_mask, return_attentions=True)
    except Exception as e:
        print(f"Error getting attentions: {e}")
        print("Your MiniDecoder may not support return_attentions parameter.")
        print("Skipping attention visualization for mini model.")
        return
    
    if attentions is None or len(attentions) == 0:
        print("No attentions returned. Skipping visualization.")
        return
        
    attn = attentions[layer_idx][0].cpu().numpy() 
    
    attn_avg = attn.mean(axis=0)[:len(tokens), :len(tokens)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn_avg,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        square=True,
        vmin=0,
        vmax=1
    )
    ax.set_title(f'Mini Model Attention Pattern (Layer {layer_idx})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap to {save_path}")


@torch.no_grad()
def visualize_attention_pretrained(
    model,
    tokenizer,
    text: str,
    config: ExperimentConfig,
    save_path: Path,
    layer_idx: int = -1
):
    print(f"\n--Generating Attention Heatmap (Pretrained Model)--")
    model.eval()
    model = model.to(config.device)
    
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(config.device)
    attention_mask = encoded['attention_mask'].to(config.device)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    valid_length = attention_mask[0].sum().item()
    tokens = tokens[:valid_length]
    
    tokens = [tok[:20] if len(tok) > 20 else tok for tok in tokens]
    
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions
    except Exception as e:
        print(f"Error getting attentions: {e}")
        print("Skipping attention visualization for pretrained model.")
        return
    
    if attentions is None or len(attentions) == 0:
        print("No attentions returned. Skipping visualization.")
        return
    
    attn = attentions[layer_idx][0].cpu().numpy()
    
    attn_avg = attn.mean(axis=0)[:len(tokens), :len(tokens)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn_avg,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        square=True,
        vmin=0,
        vmax=1
    )
    ax.set_title(f'Pretrained Model Attention Pattern (Layer {layer_idx})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap to {save_path}")


def plot_efficiency_tradeoff(
    mini_results: Dict,
    pretrained_results: Dict,
    mini_history: Dict,
    pretrained_history: Dict,
    config: ExperimentConfig
):
    print("\n--Generating Efficiency Trade-off Plots--")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    mini_acc = mini_results['accuracy'] * 100
    pretrained_acc = pretrained_results['accuracy'] * 100
    
    mini_train_time = sum(mini_history['time_per_epoch'])
    pretrained_train_time = sum(pretrained_history['time_per_epoch'])
    
    mini_inf_time = mini_results['avg_inference_time'] * 1000
    pretrained_inf_time = pretrained_results['avg_inference_time'] * 1000
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    models = ['Mini Model', 'DistilGPT-2']
    train_times = [mini_train_time / 60, pretrained_train_time / 60]
    accuracies = [mini_acc, pretrained_acc]
    colors = ['#FF6B6B', '#4ECDC4']
    
    scatter = ax1.scatter(train_times, accuracies, s=500, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax1.annotate(
            model,
            (train_times[i], accuracies[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3)
        )
    
    ax1.set_xlabel('Total Training Time (minutes)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Training Efficiency vs Accuracy', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 100])
    
    speedup = pretrained_train_time / mini_train_time
    ax1.text(
        0.05, 0.95,
        f'Training Speedup: {speedup:.2f}x',
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax2 = axes[1]
    inf_times = [mini_inf_time, pretrained_inf_time]
    
    scatter = ax2.scatter(inf_times, accuracies, s=500, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax2.annotate(
            model,
            (inf_times[i], accuracies[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3)
        )
    
    ax2.set_xlabel('Avg Inference Time per Batch (ms)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Inference Efficiency vs Accuracy', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.set_ylim([0, 100])
    
    inf_speedup = pretrained_inf_time / mini_inf_time
    ax2.text(
        0.05, 0.95,
        f'Inference Speedup: {inf_speedup:.2f}x',
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    save_path = config.output_dir / 'efficiency_tradeoff.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved efficiency trade-off plot to {save_path}")
    


def plot_comparison(mini_history: Dict, pretrained_history: Dict, config: ExperimentConfig):
    """Create comparison plots."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(mini_history['loss'], marker='o', label='Mini Model', linewidth=2)
    axes[0].plot(pretrained_history['loss'], marker='s', label='DistilGPT-2', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mini_history['accuracy'], marker='o', label='Mini Model', linewidth=2)
    axes[1].plot(pretrained_history['accuracy'], marker='s', label='DistilGPT-2', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training comparison plot to {config.output_dir / 'training_comparison.png'}")


def create_results_summary(
    mini_results: Dict,
    pretrained_results: Dict,
    mini_history: Dict,
    pretrained_history: Dict,
    config: ExperimentConfig
):
    mini_total_time = float(sum(mini_history['time_per_epoch']))
    pretrained_total_time = float(sum(pretrained_history['time_per_epoch']))
    mini_inference_time = float(mini_results['avg_inference_time'])
    pretrained_inference_time = float(pretrained_results['avg_inference_time'])
    
    summary = {
        'task': config.task,
        'mini_model': {
            'test_accuracy': float(mini_results['accuracy']),
            'avg_inference_time_per_batch': mini_inference_time,
            'final_training_loss': float(mini_history['loss'][-1]),
            'final_training_accuracy': float(mini_history['accuracy'][-1]),
            'avg_time_per_epoch': float(np.mean(mini_history['time_per_epoch'])),
            'total_training_time': mini_total_time
        },
        'pretrained_model': {
            'test_accuracy': float(pretrained_results['accuracy']),
            'avg_inference_time_per_batch': pretrained_inference_time,
            'final_training_loss': float(pretrained_history['loss'][-1]),
            'final_training_accuracy': float(pretrained_history['accuracy'][-1]),
            'avg_time_per_epoch': float(np.mean(pretrained_history['time_per_epoch'])),
            'total_training_time': pretrained_total_time
        },
        'comparison': {
            'accuracy_gap': float(pretrained_results['accuracy'] - mini_results['accuracy']),
            'training_speedup': float(pretrained_total_time / mini_total_time) if mini_total_time > 0 else 0,
            'inference_speedup': float(pretrained_inference_time / mini_inference_time) if mini_inference_time > 0 else 0
        }
    }

    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS SUMMARY - {config.task.upper()} TASK")
    print("="*70)
    print(f"\nMini Model:")
    print(f"  Test Accuracy: {mini_results['accuracy']:.4f} ({mini_results['accuracy']*100:.2f}%)")
    print(f"  Avg Inference Time: {mini_results['avg_inference_time']:.4f}s")
    print(f"  Total Training Time: {summary['mini_model']['total_training_time']:.2f}s")
    
    print(f"\nPretrained Model (DistilGPT-2):")
    print(f"  Test Accuracy: {pretrained_results['accuracy']:.4f} ({pretrained_results['accuracy']*100:.2f}%)")
    print(f"  Avg Inference Time: {pretrained_results['avg_inference_time']:.4f}s")
    print(f"  Total Training Time: {summary['pretrained_model']['total_training_time']:.2f}s")
    
    print(f"\nComparison:")
    print(f"  Accuracy Gap: {summary['comparison']['accuracy_gap']:.4f} "
          f"({abs(summary['comparison']['accuracy_gap'])*100:.2f}% {'advantage pretrained' if summary['comparison']['accuracy_gap'] > 0 else 'advantage mini'})")
    print(f"  Training Speedup (Mini vs Pretrained): {summary['comparison']['training_speedup']:.2f}x")
    print(f"  Inference Speedup (Mini vs Pretrained): {summary['comparison']['inference_speedup']:.2f}x")
    print("="*70)
    
    return summary


def run_experiment():
    """Run the complete comparison experiment."""
    set_seed(42)
    config = ExperimentConfig()
    
    print(f"Using device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    train_texts, train_labels, test_texts, test_labels, num_labels = load_nli_data(config)
    
    mini_model, mini_tokenizer = prepare_mini_model(train_texts, config, num_labels)
    mini_history = train_mini_model(mini_model, mini_tokenizer, train_texts, train_labels, config)
    mini_results = evaluate_mini_model(mini_model, mini_tokenizer, test_texts, test_labels, config)
    
    pretrained_model, pretrained_tokenizer = prepare_pretrained_model("distilgpt2", num_labels)
    pretrained_history = train_pretrained_model(pretrained_model, pretrained_tokenizer, train_texts, train_labels, config)
    pretrained_results = evaluate_pretrained_model(pretrained_model, pretrained_tokenizer, test_texts, test_labels, config)
    
    plot_comparison(mini_history, pretrained_history, config)
    
    plot_efficiency_tradeoff(mini_results, pretrained_results, mini_history, pretrained_history, config)
    
    sample_text = test_texts[0]
    
    print(f"\nVisualizing attention for sample text:")
    print(f"Text: {sample_text[:100]}...")
    
    try:
        visualize_attention_mini(
            mini_model,
            mini_tokenizer,
            sample_text,
            config,
            config.output_dir / 'attention_mini.png',
            layer_idx=-1
        )
    except Exception as e:
        print(f"Could not generate mini model attention visualization: {e}")
        print("This is likely because return_attentions is not implemented in your MiniDecoder.")
    
    try:
        visualize_attention_pretrained(
            pretrained_model,
            pretrained_tokenizer,
            sample_text,
            config,
            config.output_dir / 'attention_pretrained.png',
            layer_idx=-1
        )
    except Exception as e:
        print(f"Could not generate pretrained model attention visualization: {e}")
    
    summary = create_results_summary(mini_results, pretrained_results, mini_history, pretrained_history, config)
    
    
    return summary


if __name__ == "__main__":
    run_experiment()