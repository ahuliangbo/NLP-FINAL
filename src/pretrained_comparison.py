import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
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
import numpy as np

from models.decoder_only import DecoderConfig, MiniDecoder
from utils import SimpleTokenizer, set_seed

@dataclass
class ExperimentConfig:
    max_samples_train: int = 5000
    max_samples_test: int = 1000
    max_length: int = 256
    
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    
    mini_embed_dim: int = 128
    mini_num_layers: int = 4
    mini_num_heads: int = 4
    
    output_dir: Path = Path("outputs/part2_comparison")
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class IMDbDataset(Dataset):
    """Simple dataset wrapper for IMDb sentiment classification."""
    
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


def load_imdb_data(config: ExperimentConfig) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load and prepare IMDb dataset."""
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    train_texts = dataset['train']['text'][:config.max_samples_train]
    train_labels = dataset['train']['label'][:config.max_samples_train]
    
    test_texts = dataset['test']['text'][:config.max_samples_test]
    test_labels = dataset['test']['label'][:config.max_samples_test]
    
    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels


class MiniDecoderClassifier(nn.Module):
    """Wrap your mini decoder for sequence classification."""
    
    def __init__(self, config: DecoderConfig, num_classes: int = 2):
        super().__init__()
        self.decoder = MiniDecoder(config)
        self.classifier = nn.Linear(config.embed_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask):
        hidden = self.decoder.forward(input_ids, attention_mask)
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden.size(0)
        last_hidden = hidden[torch.arange(batch_size), sequence_lengths]
        
        logits = self.classifier(last_hidden)
        return logits


def prepare_mini_model(train_texts: List[str], config: ExperimentConfig) -> Tuple[MiniDecoderClassifier, SimpleTokenizer]:
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
        pos_type="rotary"  # Use best from Part 1
    )
    
    model = MiniDecoderClassifier(decoder_config, num_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def prepare_pretrained_model(model_name: str = "distilgpt2"):
    """Load pretrained model from Hugging Face."""
    print(f"\n--Preparing Pretrained Model: {model_name}--")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id
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
    """Train the mini model."""
    print("\n--Training Mini Model--")
    
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    dataset = IMDbDataset(train_texts, train_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    history = {'loss': [], 'accuracy': [], 'time_per_epoch': []}
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            texts = batch['text']
            labels = torch.tensor(batch['label']).to(config.device)
            
            token_ids = [tokenizer.encode(text, add_special_tokens=True)[:config.max_length] 
                        for text in texts]
            padded, attention_mask = tokenizer.pad(token_ids, max_length=config.max_length)
            input_ids = padded.to(config.device)
            attention_mask = attention_mask.to(config.device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
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
    print("\n--Fine-tuning pretrained Model--")
    
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    dataset = IMDbDataset(train_texts, train_labels, tokenizer, config.max_length)
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
        
        for batch in dataloader:
            texts = batch['text']
            labels = torch.tensor(batch['label']).to(config.device)
            
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
    model.eval()
    model = model.to(config.device)
    
    dataset = IMDbDataset(test_texts, test_labels, tokenizer, config.max_length)
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
    """Evaluate pretrained model on test set."""
    model.eval()
    model = model.to(config.device)
    
    dataset = IMDbDataset(test_texts, test_labels, tokenizer, config.max_length)
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
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }


def create_results_summary(
    mini_results: Dict,
    pretrained_results: Dict,
    mini_history: Dict,
    pretrained_history: Dict,
    config: ExperimentConfig
):
    """Create a comprehensive results summary."""
    
    summary = {
        'mini_model': {
            'test_accuracy': float(mini_results['accuracy']),  # Convert to float
            'avg_inference_time_per_batch': float(mini_results['avg_inference_time']),
            'final_training_loss': float(mini_history['loss'][-1]),
            'final_training_accuracy': float(mini_history['accuracy'][-1]),
            'avg_time_per_epoch': float(np.mean(mini_history['time_per_epoch'])),
            'total_training_time': float(sum(mini_history['time_per_epoch']))
        },
        'pretrained_model': {
            'test_accuracy': float(pretrained_results['accuracy']),
            'avg_inference_time_per_batch': float(pretrained_results['avg_inference_time']),
            'final_training_loss': float(pretrained_history['loss'][-1]),
            'final_training_accuracy': float(pretrained_history['accuracy'][-1]),
            'avg_time_per_epoch': float(np.mean(pretrained_history['time_per_epoch'])),
            'total_training_time': float(sum(pretrained_history['time_per_epoch']))
        },
        'comparison': {
            'accuracy_gap': float(pretrained_results['accuracy'] - mini_results['accuracy']),
            'speedup_ratio': float(mini_results['avg_inference_time'] / pretrained_results['avg_inference_time'])
        }
    }
    
    # Save JSON
    with open(config.output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nMini Model:")
    print(f"  Test Accuracy: {mini_results['accuracy']:.4f}")
    print(f"  Avg Inference Time: {mini_results['avg_inference_time']:.4f}s")
    print(f"  Total Training Time: {summary['mini_model']['total_training_time']:.2f}s")
    
    print(f"\nPretrained Model (DistilGPT-2):")
    print(f"  Test Accuracy: {pretrained_results['accuracy']:.4f}")
    print(f"  Avg Inference Time: {pretrained_results['avg_inference_time']:.4f}s")
    print(f"  Total Training Time: {summary['pretrained_model']['total_training_time']:.2f}s")
    
    print(f"\nComparison:")
    print(f"  Accuracy Gap: {summary['comparison']['accuracy_gap']:.4f} "
          f"({summary['comparison']['accuracy_gap']*100:.2f}% {'better' if summary['comparison']['accuracy_gap'] > 0 else 'worse'})")
    print("="*70)
    
    return summary
def run_experiment():
    """Run the complete comparison experiment."""
    set_seed(42)
    config = ExperimentConfig()
    
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(config)
    
    mini_model, mini_tokenizer = prepare_mini_model(train_texts, config)
    mini_history = train_mini_model(mini_model, mini_tokenizer, train_texts, train_labels, config)
    mini_results = evaluate_mini_model(mini_model, mini_tokenizer, test_texts, test_labels, config)
    
    pretrained_model, pretrained_tokenizer = prepare_pretrained_model("distilgpt2")
    pretrained_history = train_pretrained_model(pretrained_model, pretrained_tokenizer, train_texts, train_labels, config)
    pretrained_results = evaluate_pretrained_model(pretrained_model, pretrained_tokenizer, test_texts, test_labels, config)
    
    plot_comparison(mini_history, pretrained_history, config)
    summary = create_results_summary(mini_results, pretrained_results, mini_history, pretrained_history, config)
    
    print(f"\nAll results saved to {config.output_dir}")
    return summary


if __name__ == "__main__":
    run_experiment()