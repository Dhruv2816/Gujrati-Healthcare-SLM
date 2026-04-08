"""scripts/sample_dataset.py — Print task distribution and samples from JSONL datasets."""
import json
import random
from collections import Counter

def sample_dataset(filepath: str, n_samples: int = 5):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            
        print(f"\\n=== Dataset Stats: {filepath} ===")
        print(f"Total examples: {len(data)}")
        
        tasks = [d.get('task', 'unknown') for d in data]
        counts = Counter(tasks)
        for task, count in counts.most_common():
            print(f"- {task}: {count} ({count/len(data)*100:.1f}%)")
            
        print(f"\\n=== Random {n_samples} Samples ===")
        samples = random.sample(data, min(n_samples, len(data)))
        for i, s in enumerate(samples):
            print(f"\\n--- Sample {i+1} ({s.get('task', 'N/A')}) ---")
            messages = s.get('messages', [])
            for msg in messages:
                if msg['role'] != 'system':
                    short_content = msg['content'].replace('\\n', ' ')
                    if len(short_content) > 150:
                        short_content = short_content[:150] + "..."
                    print(f"{msg['role'].capitalize()}: {short_content}")
                    
    except Exception as e:
        print(f"❌ Failed to sample {filepath}: {e}")

if __name__ == "__main__":
    sample_dataset("data/train.jsonl")
