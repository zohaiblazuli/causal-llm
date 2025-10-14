"""Extract sample examples for the report"""
import json
from pathlib import Path

def extract_samples():
    train_file = Path("c:/isef/data/processed/train_split.jsonl")

    examples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            examples.append(json.loads(line.strip()))

    print("=== SAMPLE EXAMPLES ===\n")

    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: {ex['id']}")
        print(f"Category: {ex['task_category']}")
        print(f"Attack: {ex['attack_type']} / {ex['attack_technique']}")
        print(f"Difficulty: {ex['difficulty']}")
        print(f"\nBenign 1: {ex['user_input_benign_1'][:100]}...")
        print(f"Benign 2: {ex['user_input_benign_2'][:100]}...")
        print(f"Injection: {ex['user_input_injection'][:150]}...")
        print("-" * 80)

if __name__ == "__main__":
    extract_samples()
