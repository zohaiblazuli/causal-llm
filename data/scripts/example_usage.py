"""
Example Usage Script for Counterfactual Dataset
Demonstrates how to load and use the dataset for training
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


def load_dataset(filepath: str) -> List[Dict]:
    """Load dataset from JSONL file"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_triplet(example: Dict) -> List[Tuple[str, str]]:
    """
    Format example as triplet for causal intervention training

    Returns:
        List of (input, output) pairs for training
    """
    system = example['system_instruction']

    # Benign pair 1
    input_1 = f"{system}\n\nUser: {example['user_input_benign_1']}"
    output_1 = example['expected_output_1']

    # Benign pair 2 (counterfactual)
    input_2 = f"{system}\n\nUser: {example['user_input_benign_2']}"
    output_2 = example['expected_output_2']

    # Injection attack
    input_attack = f"{system}\n\nUser: {example['user_input_injection']}"
    output_attack = example['expected_output_injection']

    return [
        (input_1, output_1),
        (input_2, output_2),
        (input_attack, output_attack)
    ]


def format_contrastive(example: Dict) -> Dict:
    """
    Format example for contrastive learning

    Returns:
        Dictionary with anchor, positive (counterfactual), and negative (injection) examples
    """
    return {
        'anchor': {
            'system': example['system_instruction'],
            'input': example['user_input_benign_1'],
            'output': example['expected_output_1']
        },
        'positive': {  # Counterfactual - should produce similar output
            'system': example['system_instruction'],
            'input': example['user_input_benign_2'],
            'output': example['expected_output_2']
        },
        'negative': {  # Injection - should produce different output
            'system': example['system_instruction'],
            'input': example['user_input_injection'],
            'output': example['expected_output_injection']
        }
    }


def get_statistics(examples: List[Dict]) -> Dict:
    """Compute dataset statistics"""
    from collections import Counter

    stats = {
        'total': len(examples),
        'categories': Counter(ex['task_category'] for ex in examples),
        'attack_types': Counter(ex['attack_type'] for ex in examples),
        'difficulties': Counter(ex['difficulty'] for ex in examples)
    }

    return stats


def filter_by_category(examples: List[Dict], category: str) -> List[Dict]:
    """Filter examples by task category"""
    return [ex for ex in examples if ex['task_category'] == category]


def filter_by_attack_type(examples: List[Dict], attack_type: str) -> List[Dict]:
    """Filter examples by attack type"""
    return [ex for ex in examples if ex['attack_type'] == attack_type]


def filter_by_difficulty(examples: List[Dict], difficulty: str) -> List[Dict]:
    """Filter examples by difficulty level"""
    return [ex for ex in examples if ex['difficulty'] == difficulty]


def main():
    """Main demonstration"""
    print("=" * 80)
    print("COUNTERFACTUAL DATASET - USAGE EXAMPLE")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    train_data = load_dataset('data/processed/train_split.jsonl')
    val_data = load_dataset('data/processed/val_split.jsonl')
    test_data = load_dataset('data/processed/test_split.jsonl')

    print(f"   Training examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    print(f"   Test examples: {len(test_data)}")

    # Show statistics
    print("\n2. Dataset Statistics:")
    stats = get_statistics(train_data)
    print(f"   Total examples: {stats['total']}")
    print(f"\n   Categories:")
    for cat, count in stats['categories'].most_common():
        print(f"      {cat}: {count}")
    print(f"\n   Attack Types:")
    for attack, count in stats['attack_types'].most_common():
        print(f"      {attack}: {count}")

    # Example filtering
    print("\n3. Filtering Examples:")
    email_examples = filter_by_category(train_data, 'email_assistant')
    print(f"   Email assistant examples: {len(email_examples)}")

    injection_attacks = filter_by_attack_type(train_data, 'instruction_override')
    print(f"   Instruction override attacks: {len(injection_attacks)}")

    hard_examples = filter_by_difficulty(train_data, 'hard')
    print(f"   Hard difficulty examples: {len(hard_examples)}")

    # Format for training
    print("\n4. Training Format Examples:")

    # Triplet format
    print("\n   a) Triplet Format (for causal intervention training):")
    example = random.choice(train_data)
    triplet = format_triplet(example)

    print(f"\n      Example ID: {example['id']}")
    print(f"      Category: {example['task_category']}")
    print(f"      Attack Type: {example['attack_type']}")

    for i, (input_text, output_text) in enumerate(triplet, 1):
        print(f"\n      Pair {i}:")
        print(f"         Input:  {input_text[:100]}...")
        print(f"         Output: {output_text[:100]}...")

    # Contrastive format
    print("\n   b) Contrastive Format (for contrastive learning):")
    example = random.choice(train_data)
    contrastive = format_contrastive(example)

    print(f"\n      Example ID: {example['id']}")
    print(f"\n      Anchor (Benign):")
    print(f"         Input:  {contrastive['anchor']['input'][:80]}...")
    print(f"         Output: {contrastive['anchor']['output'][:80]}...")

    print(f"\n      Positive (Counterfactual):")
    print(f"         Input:  {contrastive['positive']['input'][:80]}...")
    print(f"         Output: {contrastive['positive']['output'][:80]}...")

    print(f"\n      Negative (Injection):")
    print(f"         Input:  {contrastive['negative']['input'][:80]}...")
    print(f"         Output: {contrastive['negative']['output'][:80]}...")

    # PyTorch DataLoader example
    print("\n5. PyTorch Integration Example:")
    print("""
    from torch.utils.data import Dataset, DataLoader

    class CounterfactualDataset(Dataset):
        def __init__(self, filepath):
            self.examples = load_dataset(filepath)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return format_triplet(self.examples[idx])

    # Create DataLoader
    train_dataset = CounterfactualDataset('data/processed/train_split.jsonl')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for batch in train_loader:
        # batch contains triplets of (input, output) pairs
        # Implement your training logic here
        pass
    """)

    print("\n" + "=" * 80)
    print("For more details, see data/README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
