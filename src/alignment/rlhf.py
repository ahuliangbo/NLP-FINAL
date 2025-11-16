"""Starter code for the lightweight RLHF / DPO alignment utilities."""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from assignment4.models.decoder_only import MiniDecoder
from assignment4.utils import load_json, pad_sequences


class PreferenceDataset:
    """Dataset producing (chosen, rejected) response pairs for alignment."""

    def __init__(self, tokenizer, path: str, seed: int = 42):
        self.tokenizer = tokenizer
        self.data = load_json(path)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.data)

    def sample_batch(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        indices = torch.randint(0, len(self.data), (batch_size,), generator=self.rng)
        
        chosen_ids = []
        rejected_ids = []
        
        for idx in indices:
            item = self.data[int(idx)]
            chosen_ids.append(self.tokenizer.encode(item["prompt"] + " " + item["chosen"]))
            rejected_ids.append(self.tokenizer.encode(item["prompt"] + " " + item["rejected"]))
        
        chosen_padded, chosen_mask = pad_sequences(chosen_ids, self.tokenizer.token_id(self.tokenizer.pad_token))
        rejected_padded, rejected_mask = pad_sequences(rejected_ids, self.tokenizer.token_id(self.tokenizer.pad_token))
        
        prompt_lens = [len(self.tokenizer.encode(self.data[int(idx)]["prompt"])) for idx in indices]
        chosen_response_mask = torch.zeros_like(chosen_mask)
        rejected_response_mask = torch.zeros_like(rejected_mask)
        
        for i, plen in enumerate(prompt_lens):
            chosen_response_mask[i, plen:] = chosen_mask[i, plen:]
            rejected_response_mask[i, plen:] = rejected_mask[i, plen:]
        
        return {
            "chosen_input_ids": chosen_padded.to(device),
            "chosen_attention_mask": chosen_mask.to(device),
            "chosen_response_mask": chosen_response_mask.to(device),
            "rejected_input_ids": rejected_padded.to(device),
            "rejected_attention_mask": rejected_mask.to(device),
            "rejected_response_mask": rejected_response_mask.to(device),
        }


class RewardModel(nn.Module):
    """Small reward head that scores decoder hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, response_mask: Tensor | None = None) -> Tensor:
        mask = response_mask if response_mask is not None else attention_mask
        scores = self.network(hidden_states).squeeze(-1)
        masked_scores = scores * mask
        pooled = masked_scores.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled


@dataclass
class PolicyLossOutput:
    """Container for the policy loss scalar and logging metrics."""

    loss: Tensor
    metrics: Dict[str, float]


def compute_policy_loss(
    model: MiniDecoder,
    ref_model: MiniDecoder,
    reward_model: RewardModel,
    batch: Dict[str, Tensor],
    method: str = "dpo",
    beta: float = 0.1,
) -> PolicyLossOutput:
    if method == "dpo":
        policy_chosen_lp = model.sequence_log_probs(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"]
        )
        policy_rejected_lp = model.sequence_log_probs(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"]
        )
        
        with torch.no_grad():
            ref_chosen_lp = ref_model.sequence_log_probs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_mask"]
            )
            ref_rejected_lp = ref_model.sequence_log_probs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"]
            )
            
            chosen_hidden = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected_hidden = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            chosen_rewards = reward_model(chosen_hidden, batch["chosen_attention_mask"], batch["chosen_response_mask"])
            rejected_rewards = reward_model(rejected_hidden, batch["rejected_attention_mask"], batch["rejected_response_mask"])
        
        logits = (policy_chosen_lp - policy_rejected_lp) - (ref_chosen_lp - ref_rejected_lp)
        loss = -F.logsigmoid(logits / beta).mean()
        
        metrics = {
            "loss": loss.item(),
            "chosen_reward": (policy_chosen_lp - ref_chosen_lp).mean().item(),
            "rejected_reward": (policy_rejected_lp - ref_rejected_lp).mean().item(),
            "avg_reward": (chosen_rewards - rejected_rewards).mean().item(),
            "pref_acc": (logits > 0).float().mean().item(),
        }
    else:
        chosen_hidden = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected_hidden = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        
        chosen_rewards = reward_model(chosen_hidden, batch["chosen_attention_mask"], batch["chosen_response_mask"])
        rejected_rewards = reward_model(rejected_hidden, batch["rejected_attention_mask"], batch["rejected_response_mask"])
        
        policy_chosen_lp = model.sequence_log_probs(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"]
        )
        policy_rejected_lp = model.sequence_log_probs(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"]
        )
        
        with torch.no_grad():
            ref_chosen_lp = ref_model.sequence_log_probs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_mask"]
            )
            ref_rejected_lp = ref_model.sequence_log_probs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"]
            )
        
        chosen_kl = policy_chosen_lp - ref_chosen_lp
        rejected_kl = policy_rejected_lp - ref_rejected_lp
        
        loss = -(chosen_rewards - beta * chosen_kl - (rejected_rewards - beta * rejected_kl)).mean()
        
        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "kl": chosen_kl.mean().item(),
            "pref_acc": (chosen_rewards > rejected_rewards).float().mean().item(),
        }
    
    return PolicyLossOutput(loss=loss, metrics=metrics)

def rlhf_step(
    model: MiniDecoder,
    ref_model: MiniDecoder,
    reward_model: RewardModel,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
    method: str = "dpo",
    beta: float = 0.1,
    clip_norm: float = 1.0,
) -> Dict[str, float]:
    optimizer.zero_grad()
    output = compute_policy_loss(model, ref_model, reward_model, batch, method, beta)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if method == "ppo":
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), clip_norm)
    optimizer.step()
    return output.metrics