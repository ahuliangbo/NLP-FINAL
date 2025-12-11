from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from utils import PositionalEncoding, ResidualBlock, build_causal_mask, build_padding_mask


class CausalSelfAttentionBlock(nn.Module):
    """Causal self-attention block used inside :class:`MiniDecoder`."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float, rotary: bool = False):
        super().__init__()
        self.block = ResidualBlock(embed_dim, num_heads, ff_dim, dropout, rotary=rotary)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, return_attention_weights: bool = False):
        """Apply left-to-right self-attention.

        Args:
            hidden_states: Tensor `(batch, seq_len, embed_dim)`.
            attention_mask: Tensor `(batch, seq_len)` with padding markers.
            return_attention_weights: If True, return (output, attention_weights).

        Returns:
            Tensor `(batch, seq_len, embed_dim)` after causal attention.
            If return_attention_weights=True, returns tuple (output, attn_weights).
        """
        causal_mask = build_causal_mask(hidden_states.size(1), device=hidden_states.device)
        key_padding_mask = build_padding_mask(attention_mask)
        
        if return_attention_weights:
            output, attn_weights = self.block(
                hidden_states, 
                attn_mask=causal_mask, 
                key_padding_mask=key_padding_mask,
                return_attention_weights=True
            )
            return output, attn_weights
        else:
            return self.block(hidden_states, attn_mask=causal_mask, key_padding_mask=key_padding_mask)


@dataclass
class DecoderConfig:
    """Configuration for :class:`MiniDecoder`."""

    vocab_size: int
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 4
    max_seq_len: int = 64
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pos_type: str = "sinusoidal"  # options: "sinusoidal", "learned", "rotary"


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding with extrapolation support."""
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        device = x.device
        
        if seq_len <= self.max_seq_len:
            # Normal case: within trained range
            pos = self.emb(torch.arange(seq_len, device=device))
            return x + pos.unsqueeze(0)
        else:
            # Extrapolation: repeat last position for tokens beyond max_seq_len
            positions = torch.arange(seq_len, device=device)
            # Clamp positions to max_seq_len - 1
            positions = positions.clamp(max=self.max_seq_len - 1)
            pos = self.emb(positions)
            return x + pos.unsqueeze(0)


class MiniDecoder(nn.Module):
    """Decoder-only transformer for causal language modelling."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        
        # Positional encoding selection
        pos_type = getattr(config, "pos_type", "sinusoidal")
        if pos_type == "sinusoidal":
            self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_seq_len)
        elif pos_type == "learned":
            # Use the fixed learned positional encoding class
            self.pos_encoding = LearnedPositionalEncoding(config.max_seq_len, config.embed_dim)
        else:
            # For rotary, attention handles positions; keep identity here
            self.pos_encoding = nn.Identity()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                CausalSelfAttentionBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.ff_dim,
                    config.dropout,
                    rotary=(pos_type == "rotary"),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, return_attentions: bool = False):
        """Compute hidden states for a batch of prefix sequences.

        Args:
            input_ids: Tensor `(batch, seq_len)` with token ids.
            attention_mask: Tensor `(batch, seq_len)` marking valid tokens.
            return_attentions: If True, return (hidden_states, list_of_attention_weights).

        Returns:
            Tensor `(batch, seq_len, embed_dim)` of decoder hidden states.
            If return_attentions=True, returns tuple (hidden_states, list_of_attention_weights).
        """
        x = self.token_embed(input_ids)  # FIXED: was self.embedding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        all_attentions = [] if return_attentions else None
        
        for layer in self.layers:
            if return_attentions:
                x, attn_weights = layer(x, attention_mask, return_attention_weights=True)
                all_attentions.append(attn_weights)
            else:
                x = layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        if return_attentions:
            return x, all_attentions
        return x

    def logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Convenience helper returning vocabulary logits.

        Args:
            input_ids: Tensor `(batch, seq_len)`.
            attention_mask: Tensor `(batch, seq_len)`.

        Returns:
            Logits `(batch, seq_len, vocab_size)`.
        """
        hidden_states = self.forward(input_ids, attention_mask)
        return self.lm_head(hidden_states)

    def forward_train(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Teacher-forced training forward pass.

        Args:
            input_ids: Tensor `(batch, seq_len)` containing BOS + targets.
            attention_mask: Optional tensor `(batch, seq_len)` marking valid tokens.
            labels: Optional tensor `(batch, seq_len)` for next-token prediction.

        Returns:
            Dictionary containing at least keys `loss`, `logits`, `accuracy`.
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if labels is None:
            labels = input_ids
        
        logits = self.logits(input_ids, attention_mask)
        loss_mask = (labels != self.config.pad_token_id)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), reduction='none')
        loss = (loss * loss_mask.view(-1).float()).sum() / loss_mask.sum()
        
        preds = logits.argmax(dim=-1)
        accuracy = ((preds == labels) & loss_mask).float().sum() / loss_mask.sum()
        
        return {"loss": loss, "logits": logits, "accuracy": accuracy}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> Tensor:
        generated = input_ids
        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            if finished.all():
                break
                
            attention_mask = (generated != self.config.pad_token_id).long()
            logits = self.logits(generated, attention_mask)
            next_token_logits = logits[:, -1, :]
            
            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            finished = finished | (next_token.squeeze(-1) == self.config.eos_token_id)
            generated = torch.cat([generated, next_token], dim=1)
            
            if finished.all():
                break
        
        return generated

    def sequence_log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = self.logits(input_ids, attention_mask)
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        if target_mask is None:
            target_mask = attention_mask[:, 1:]
        else:
            target_mask = target_mask[:, 1:]
        
        return (token_log_probs * target_mask.float()).sum(dim=1)

def build_few_shot_prompt(examples: List[Dict[str, str]], query: str) -> str:
    """Helper used in the training script to form a few-shot prompt."""
    prompt_lines = []
    for example in examples:
        prompt_lines.append(f"Input: {example['input']}")
        prompt_lines.append(f"Output: {example['output']}")
    prompt_lines.append(f"Input: {query}")
    prompt_lines.append("Output:")
    return "\n".join(prompt_lines)