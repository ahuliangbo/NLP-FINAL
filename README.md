# Mini Transformer Training

A PyTorch implementation of a compact decoder-only transformer for causal language modeling.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Train on Pattern Sequences (Default)

```bash
cd src
python -m train_decoder --epochs 20
```

### Option 2: Train on WikiText-2

#### Step 1: Download and prepare WikiText-2
```bash
cd src
python download_wikitext2.py --output data/wikitext2.txt --max-lines 10000
```

This downloads the WikiText-2 training split and saves the first 10,000 lines.

#### Step 2: Train on WikiText-2
```bash
python -m train_decoder --data-path data/wikitext2.txt --epochs 20 --batch-size 16
```

## Usage Examples

### Training with custom parameters

```bash
python -m train_decoder \
  --data-path data/wikitext2.txt \
  --epochs 30 \
  --batch-size 32 \
  --embed-dim 256 \
  --num-heads 8 \
  --num-layers 6 \
  --dropout 0.2 \
  --lr 1e-4 \
  --max-lines 50000
```

### Using a small dataset for quick testing

```bash
python -m train_decoder --epochs 5 --batch-size 4 --max-lines 100
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/pattern_sequences.txt` | Path to training data (text file, one sequence per line) |
| `--output-dir` | `checkpoints` | Directory to save model checkpoint |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 8 | Training batch size |
| `--embed-dim` | 128 | Embedding dimension |
| `--num-heads` | 4 | Number of attention heads |
| `--ff-dim` | 256 | Feed-forward network dimension |
| `--num-layers` | 4 | Number of transformer layers |
| `--dropout` | 0.1 | Dropout probability |
| `--max-seq-len` | 64 | Maximum sequence length |
| `--lr` | 3e-4 | Learning rate |
| `--clip-norm` | 1.0 | Gradient clipping norm |
| `--max-lines` | None | Limit number of sequences to load |
| `--seed` | 42 | Random seed for reproducibility |

## Download WikiText-2 Arguments

```bash
python download_wikitext2.py [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `data/wikitext2.txt` | Output file path |
| `--split` | `train` | Dataset split (train, validation, test) |
| `--max-lines` | None | Maximum lines to save (None = all) |

## Model Architecture

- **Type**: Decoder-only transformer (causal LM)
- **Positional Encoding**: Sinusoidal
- **Attention**: Causal multi-head self-attention
- **Feed-forward**: Position-wise MLP with GELU activation
- **Normalization**: Layer normalization

## Output

Training saves a checkpoint at `checkpoints/decoder.pt` containing:
- `model_state`: Model weights and parameters
- `tokenizer_vocab`: Tokenizer vocabulary mapping

## Example: Load and Inference

```python
import torch
from src.models.decoder_only import DecoderConfig, MiniDecoder
from src.utils import SimpleTokenizer

# Load checkpoint
checkpoint = torch.load("checkpoints/decoder.pt", map_location="cpu")
tokenizer_vocab = checkpoint["tokenizer_vocab"]

# Recreate tokenizer
tokenizer = SimpleTokenizer([""])
tokenizer.token_to_id = tokenizer_vocab
tokenizer.id_to_token = {idx: tok for tok, idx in tokenizer_vocab.items()}

# Load model
config = DecoderConfig(vocab_size=len(tokenizer_vocab))
model = MiniDecoder(config)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Generate text
prompt = "red blue"
tokens = tokenizer.encode(prompt, add_special_tokens=True)
generated = model.generate(torch.tensor([tokens]), max_new_tokens=10, temperature=0.7)
output = tokenizer.decode(generated[0].tolist())
print(output)
```

## Dataset Formats

### Text Format (Default)
Plain text file with one sequence per line:
```
red blue red blue red
one two three one two
alpha beta gamma alpha beta
```

### Prepare Custom Data

Any text file with one sequence/document per line will work:
```bash
python -m train_decoder --data-path your_data.txt
```

## Notes

- WikiText-2 typically contains longer documents; adjust `--max-seq-len` as needed
- For very large datasets, use `--max-lines` to limit memory usage during vocab building
- Consider increasing `--embed-dim` and `--num-layers` for better performance on larger datasets
- Default batch size works for most GPUs; increase for faster training if you have memory

