"""Training script for the decoder-only causal language model."""

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from models.decoder_only import DecoderConfig, MiniDecoder, build_few_shot_prompt
from utils import SimpleTokenizer, set_seed

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_DIR / "data" / "pattern_sequences.txt"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "checkpoints"


class PatternDataset(Dataset):
    """Dataset made from repeating pattern sequences."""

    def __init__(self, sequences: List[str], tokenizer: SimpleTokenizer, max_length: int = None):
        """Store tokenized pattern sequences.
        
        Args:
            sequences: List of text sequences.
            tokenizer: Tokenizer for encoding.
            max_length: Maximum sequence length (in tokens). Longer sequences are truncated.
        """
        self.sequences = [seq for seq in sequences if seq.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, index: int):
        """Tokenize one sequence and add BOS/EOS tokens."""
        tokens = self.tokenizer.encode(self.sequences[index], add_special_tokens=True)
        # Truncate to max_length if specified
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return {"input_ids": tokens}


def collate_fn(batch, tokenizer: SimpleTokenizer):
    """Pad sequences for training.

    Args:
        batch: List of dictionaries produced by `PatternDataset`.
        tokenizer: Tokenizer used for padding.

    Returns:
        Dictionary containing padded ids, attention mask, and labels.
    """
    input_ids = [item["input_ids"] for item in batch]
    padded, attention = tokenizer.pad(input_ids)
    return {
        "input_ids": padded,
        "attention_mask": attention,
        "labels": padded.clone(),
    }


def load_sequences(path: Path, max_lines: int = None) -> List[str]:
    """Read newline-separated sequences from disk.

    Args:
        path: Path to the text file containing sequences.
        max_lines: Optional limit on number of lines to load.

    Returns:
        List of stripped sequence strings.
    """
    sequences = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_lines and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                sequences.append(stripped)
    return sequences


def train(args):
    """Main training loop for the decoder-only model.

    Args:
        args: Parsed CLI arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from {args.data_path}...")
    sequences = load_sequences(args.data_path, max_lines=args.max_lines)
    print(f"Loaded {len(sequences)} sequences")
    
    tokenizer = SimpleTokenizer(sequences)
    dataset = PatternDataset(sequences, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    config = DecoderConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.token_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_id(tokenizer.eos_token),
    )

    model = MiniDecoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.forward_train(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_acc += float(outputs["accuracy"].item())
            steps += 1
        avg_loss = epoch_loss / max(1, steps)
        avg_acc = epoch_acc / max(1, steps)
        print(f"Epoch {epoch:02d} - loss: {avg_loss:.4f} - token_acc: {avg_acc:.4f}")

    prompt = build_few_shot_prompt(
        examples=[
            {"input": sequences[0], "output": sequences[0].split()[-1]},
            {"input": sequences[1], "output": sequences[1].split()[-1]},
        ],
        query="red blue red blue",
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    generated = model.generate(torch.tensor([tokens], device=device), max_new_tokens=5)
    prediction = tokenizer.decode(generated[0].tolist())
    print("Few-shot prompt:\n", prompt)
    print("Generated continuation:\n", prediction)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer_vocab": tokenizer.token_to_id,
        },
        Path(args.output_dir) / "decoder.pt",
    )


def parse_args():
    """Parse command-line flags for decoder training.

    Returns:
        Namespace of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the decoder-only language model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length (longer sequences will be truncated)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lines", type=int, default=None, help="Limit number of lines to load from dataset")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-query", type=str, default="red blue red blue")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)
