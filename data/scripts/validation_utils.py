"""
Validation Utilities - Week 1 Phase 2
Common utility functions used across validation scripts.
"""

import json
from pathlib import Path


def load_jsonl(filepath):
    """
    Load JSONL file and return list of examples.

    Args:
        filepath: Path to JSONL file (str or Path)

    Returns:
        List of dictionaries
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    return examples


def save_json(data, filepath):
    """
    Save data to JSON file with pretty printing.

    Args:
        data: Data to save
        filepath: Path to save to (str or Path)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(examples, filepath):
    """
    Save examples to JSONL file.

    Args:
        examples: List of dictionaries
        filepath: Path to save to (str or Path)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def get_dataset_paths():
    """
    Get standard dataset paths.

    Returns:
        Dictionary with paths to train/val/test splits
    """
    base_dir = Path("C:/isef")

    return {
        "train": base_dir / "data/processed/train_split.jsonl",
        "val": base_dir / "data/processed/val_split.jsonl",
        "test": base_dir / "data/processed/test_split.jsonl",
        "processed_dir": base_dir / "data/processed"
    }


def print_section_header(title, width=70):
    """Print a formatted section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_subsection(title, width=70):
    """Print a formatted subsection header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width + "\n")


def format_percentage(value, total):
    """Format a value as a percentage."""
    if total == 0:
        return "0.0%"
    return f"{value/total*100:.1f}%"


def format_bar(percentage, max_width=40):
    """
    Create a visual bar for percentages.

    Args:
        percentage: Percentage value (0-100)
        max_width: Maximum width of bar

    Returns:
        String with bar representation
    """
    bar_length = int(percentage / 100 * max_width)
    return "#" * bar_length


if __name__ == "__main__":
    # Test utilities
    print("Validation Utilities Test")
    print_section_header("Testing utilities")

    paths = get_dataset_paths()
    print("Dataset paths:")
    for name, path in paths.items():
        exists = path.exists() if isinstance(path, Path) else False
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  {name:15s}: {status}")

    print("\nTest passed!")
