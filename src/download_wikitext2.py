"""Download and preprocess WikiText-2 dataset."""

import argparse
from pathlib import Path
from datasets import load_dataset


def download_wikitext2(output_path: Path, split: str = "train", max_lines: int = None):
    """Download WikiText-2 and save to text file.
    
    Args:
        output_path: Path to save the processed dataset.
        split: Which split to download ('train', 'validation', 'test').
        max_lines: Optional limit on number of lines to save.
    """
    print(f"Downloading WikiText-2 {split} split...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
    
    print(f"Processing {len(dataset)} examples...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    line_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            # Skip empty lines and metadata markers (=, -, etc.)
            if text and not text.startswith("="):
                # Split long documents into sentences/paragraphs for better tokenization
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        f.write(line + "\n")
                        line_count += 1
                        if max_lines and line_count >= max_lines:
                            break
            if max_lines and line_count >= max_lines:
                break
    
    print(f"Saved {line_count} lines to {output_path}")


def main():
    """Parse arguments and download dataset."""
    parser = argparse.ArgumentParser(description="Download WikiText-2 dataset")
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path(__file__).parent / "data" / "wikitext2.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split to download"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines to save (default: all)"
    )
    
    args = parser.parse_args()
    download_wikitext2(args.output, args.split, args.max_lines)


if __name__ == "__main__":
    main()
