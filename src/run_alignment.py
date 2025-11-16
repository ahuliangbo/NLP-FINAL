"""Command-line driver for the lightweight RLHF alignment loop."""

import argparse
from pathlib import Path

import torch

from assignment4.alignment.rlhf import PreferenceDataset, RewardModel, rlhf_step
from assignment4.models.decoder_only import DecoderConfig, MiniDecoder
from assignment4.utils import SimpleTokenizer, load_json, set_seed

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_DIR / "data" / "synthetic_intents.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "checkpoints"


def build_tokenizer(records):
    """Construct a tokenizer vocabulary using prompts and responses.

    Args:
        records: Iterable of dictionaries containing `utterance` and `intent`.

    Returns:
        `SimpleTokenizer` fitted to the dataset.
    """
    texts = []
    for record in records:
        utterance = record["utterance"]
        intent = record["intent"]
        texts.append(f"User: {utterance}\nAssistant:")
        texts.append(f"The intent is {intent}.")
    return SimpleTokenizer(texts)


def run(args):
    """Execute the RLHF fine-tuning loop.

    Args:
        args: Parsed CLI arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    records = load_json(str(args.data_path))
    tokenizer = build_tokenizer(records)

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

    policy = MiniDecoder(config).to(device)
    reference = MiniDecoder(config).to(device)
    reference.load_state_dict(policy.state_dict())
    for param in reference.parameters():
        param.requires_grad = False

    reward_model = RewardModel(config.embed_dim).to(device)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(reward_model.parameters()), lr=args.lr)
    dataset = PreferenceDataset(tokenizer, str(args.data_path))

    for step in range(1, args.steps + 1):
        batch = dataset.sample_batch(args.batch_size, device=device)
        metrics = rlhf_step(
            policy,
            reference,
            reward_model,
            optimizer,
            batch,
            method=args.method,
            beta=args.beta,
            clip_norm=args.clip_norm,
        )
        if step % args.log_every == 0 or step == 1:
            print(
                f"Step {step:04d}: "
                f"loss={metrics['loss']:.4f} "
                f"reward={metrics['avg_reward']:.4f} "
                f"kl={metrics['kl_to_ref']:.4f} "
                f"pref_acc={metrics['pref_acc']:.4f}"
            )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "reward_state": reward_model.state_dict(),
            "tokenizer_vocab": tokenizer.token_to_id,
        },
        Path(args.output_dir) / "rlhf_policy.pt",
    )


def parse_args():
    """Parse CLI options for the alignment run.

    Returns:
        Namespace with argument values.
    """
    parser = argparse.ArgumentParser(description="Run lightweight RLHF alignment.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "ppo"])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run(args)
