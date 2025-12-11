"""Shared utilities for Assignment 4 transformer models.

This module provides lightweight tokenization, batching helpers, and common
transformer building blocks that are reused across the encoder, decoder, and
seq2seq models. The goal is to keep the implementation accessible for
students while mirroring the components found in larger frameworks such as
PyTorch's `nn.Transformer`.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>"]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility.

    Args:
        seed: Integer seed shared across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleTokenizer:
    """Whitespace tokenizer with a fixed vocabulary and special tokens."""

    def __init__(self, texts: Iterable[str], extra_tokens: Optional[Iterable[str]] = None):
        """Build a vocabulary from the provided corpus.

        Args:
            texts: Corpus used to derive the vocabulary.
            extra_tokens: Optional iterable of additional tokens to insert.
        """
        vocab = list(SPECIAL_TOKENS)
        if extra_tokens:
            vocab.extend(extra_tokens)
        seen = set(vocab)
        for text in texts:
            for token in text.strip().split():
                if token not in seen:
                    vocab.append(token)
                    seen.add(token)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.mask_token = "<mask>"

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.token_to_id)

    def token_id(self, token: str) -> int:
        """Map a token to its integer id."""
        return self.token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token ids.

        Args:
            text: Input string to tokenize.
            add_special_tokens: If True, prepend `<bos>` and append `<eos>`.

        Returns:
            List of token ids.
        """
        tokens = text.strip().split()
        ids = [self.token_to_id.get(token, self.token_to_id[self.mask_token]) for token in tokens]
        if add_special_tokens:
            return [self.token_id(self.bos_token)] + ids + [self.token_id(self.eos_token)]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert ids back into text.

        Args:
            ids: Token ids to decode.
            skip_special_tokens: Whether to drop BOS/EOS/MASK/PAD tokens.

        Returns:
            Detokenized string.
        """
        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.mask_token)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def pad(self, sequences: Sequence[Sequence[int]], max_length: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Pad a batch of token sequences.

        Args:
            sequences: Batch of variable-length token sequences.
            max_length: Optional maximum length to pad/truncate to.

        Returns:
            Tuple of `(padded_ids, attention_mask)` tensors.
        """
        pad_id = self.token_id(self.pad_token)
        return pad_sequences(sequences, pad_value=pad_id, max_length=max_length)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_value: int = 0,
    max_length: Optional[int] = None,
    dtype: torch.dtype = torch.long,
) -> Tuple[Tensor, Tensor]:
    """Pad / truncate sequences to a uniform length.

    Args:
        sequences: Sequence batch to pad.
        pad_value: Value used to pad positions beyond sequence length.
        max_length: Optional target sequence length.
        dtype: Tensor dtype for the padded output.

    Returns:
        Tuple `(padded, attention_mask)` where the mask contains `1` for real
        tokens and `0` for padded positions.
    """
    if not sequences:
        raise ValueError("pad_sequences expects a non-empty sequence.")
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if length > 0:
            padded[i, :length] = torch.tensor(seq[:length], dtype=dtype)
    attention_mask = (padded != pad_value).long()
    return padded, attention_mask


def build_causal_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """Construct an upper-triangular mask used for causal self-attention.

    Args:
        size: Sequence length.
        device: Optional target device for the mask.

    Returns:
        Attention mask tensor with `-inf` above the diagonal.
    """
    mask = torch.triu(torch.full((size, size), float("-inf")), diagonal=1)
    if device:
        mask = mask.to(device)
    return mask


def build_padding_mask(attention_mask: Tensor) -> Tensor:
    """Convert a binary attention mask to a boolean padding mask.

    Args:
        attention_mask: Tensor containing 1 for tokens and 0 for padding.

    Returns:
        Boolean tensor where `True` marks padding positions.
    """
    # Expected shape (batch, seq_len); helps guard against silent shape bugs.
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be 2D")
    return attention_mask == 0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dynamic extension for longer sequences."""

    def __init__(self, embed_dim: int, max_len: int = 2048):
        """Pre-compute sinusoidal encodings per `Attention Is All You Need`.

        Args:
            embed_dim: Transformer embedding dimension.
            max_len: Initial maximum sequence length supported.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = self._generate_positional_encoding(max_len, embed_dim)
        self.register_buffer("pe", pe, persistent=False)

    def _generate_positional_encoding(self, max_len: int, embed_dim: int) -> Tensor:
        """Generate sinusoidal positional encodings.
        
        Args:
            max_len: Maximum sequence length.
            embed_dim: Embedding dimension.
            
        Returns:
            Positional encoding tensor of shape (1, max_len, embed_dim).
        """
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to the input embeddings.

        Args:
            x: Input embeddings of shape `(batch, seq_len, embed_dim)`.

        Returns:
            Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        
        # If sequence is longer than current buffer, extend it
        if seq_len > self.pe.size(1):
            # Generate new positional encodings up to seq_len
            new_pe = self._generate_positional_encoding(seq_len, self.embed_dim)
            # Move to same device as input
            self.pe = new_pe.to(x.device)
            self.max_len = seq_len
        
        return x + self.pe[:, :seq_len]

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float):
        """Create the hidden MLP block used in transformer layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network.

        Args:
            x: Hidden states of shape `(batch, seq_len, embed_dim)`.

        Returns:
            Transformed tensor of the same shape.
        """
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting self- and cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialise a multi-head attention layer.

        Args:
            embed_dim: Transformer embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout applied to attention weights and outputs.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention_weights: bool = False,  # NEW PARAMETER
    ) -> Tensor:
        """Perform attention computation.

        Args:
            query: Query tensor (batch, seq_q, embed_dim).
            key: Key tensor (batch, seq_k, embed_dim).
            value: Value tensor (batch, seq_k, embed_dim).
            attn_mask: Optional additive mask applied to attention logits.
            key_padding_mask: Boolean mask for padded key/value positions.
            return_attention_weights: If True, return (output, attention_weights).

        Returns:
            Attention output of shape `(batch, seq_q, embed_dim)`.
            If return_attention_weights=True, returns tuple (output, attn_weights).
        """
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn)
        output = torch.matmul(attn_dropped, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(output)
        
        if return_attention_weights:
            return output, attn  # Return pre-dropout attention weights
        return output


class RotaryMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with Rotary Position Embeddings (RoPE) applied to queries and keys."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)

    def _build_sin_cos(self, seq_len: int, device: Optional[torch.device] = None):
        # Build sin and cos for RoPE
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("n,d->nd", positions, inv_freq)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        return sin, cos

    def _apply_rope(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # x shape: (batch, heads, seq_len, head_dim)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot_1 = x1 * cos.unsqueeze(0).unsqueeze(0) - x2 * sin.unsqueeze(0).unsqueeze(0)
        x_rot_2 = x1 * sin.unsqueeze(0).unsqueeze(0) + x2 * cos.unsqueeze(0).unsqueeze(0)
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rot_1
        x_out[..., 1::2] = x_rot_2
        return x_out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention_weights: bool = False,  # NEW PARAMETER
    ) -> Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        seq_len = max(query_len, key_len)
        sin, cos = self._build_sin_cos(seq_len, device=query.device)

        q = self._apply_rope(q, sin[:query_len], cos[:query_len])
        k = self._apply_rope(k, sin[:key_len], cos[:key_len])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn)
        output = torch.matmul(attn_dropped, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(output)
        
        if return_attention_weights:
            return output, attn  # Return pre-dropout attention weights
        return output


class ResidualBlock(nn.Module):
    """Transformer-style residual block with attention and feed-forward."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float, rotary: bool = False):
        """Create a residual attention + MLP block."""
        super().__init__()
        if rotary:
            self.self_attn = RotaryMultiHeadAttention(embed_dim, num_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        return_attention_weights: bool = False,  # NEW PARAMETER
    ) -> Tensor:
        """Apply attention, residual connection, and feed-forward network.

        Args:
            x: Query tensor.
            attn_mask: Optional additive mask for attention logits.
            key_padding_mask: Boolean mask for padded positions.
            kv: Optional key/value tensor for cross-attention.
            return_attention_weights: If True, return (output, attention_weights).

        Returns:
            Tensor after attention + FFN processing.
            If return_attention_weights=True, returns tuple (output, attn_weights).
        """
        # Use provided key/value tensor for cross-attention, default to self-attn.
        key_value = x if kv is None else kv
        
        if return_attention_weights:
            attn_out, attn_weights = self.self_attn(
                x, key_value, key_value, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask,
                return_attention_weights=True
            )
        else:
            attn_out = self.self_attn(
                x, key_value, key_value, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask
            )
            attn_weights = None
            
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        if return_attention_weights:
            return x, attn_weights
        return x

def load_json(path: str) -> List[Dict[str, str]]:
    """Load a JSON file containing a list of dictionaries.

    Args:
        path: Path to the JSON file on disk.

    Returns:
        Parsed list of dictionaries.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Batch:
    """Lightweight batch container used by training utilities."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Optional[Tensor] = None
    aux: Optional[Dict[str, Tensor]] = None
