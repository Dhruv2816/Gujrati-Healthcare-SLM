import os
import json
import torch
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Ignore Deprecation Warnings
warnings.filterwarnings("ignore")

print("="*60)
print("🏗️ GUJARATI HEALTHCARE SLM - BASE MODEL BENCHMARK 🏗️")
print("="*60)

# 1. Load Data
test_file = "data/test.jsonl"
print(f"📂 Loading Test Data from {test_file}...")

test_cases = []
if os.path.exists(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 15: # Test on first 15 queries to compare fairly with fine-tuned
                break
                
            data = json.loads(line)
            user_msg = next(m["content"] for m in data["messages"] if m["role"] == "user")
            asst_msg = next(m["content"] for m in data["messages"] if m["role"] == "assistant")
            test_cases.append((user_msg, asst_msg))
else:
    print(f"❌ {test_file} not found. Cannot evaluate.")
    exit()

print(f"✅ Loaded {len(test_cases)} samples.")

# 2. Load Base Model on CPU
base_id = "Qwen/Qwen2.5-3B-Instruct"

print("\n⚙️ Loading Base Model to CPU (This will take System RAM)...")
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="cpu", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

print(f"✅ BASE MODEL ONLY loaded (No fine-tuned weights attached!)")

model.eval()

# 3. Generate Predictions
print("\n🤖 Generating Predictions (Slow, running on CPU)...")

predictions = []
references = []

for instruction, target in tqdm(test_cases):
    messages = [
        {"role": "system", "content": "You are a safe, accurate Gujarati healthcare assistant."},
        {"role": "user", "content": instruction}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False, # Guaranteed Greedy Decoding for deterministic scores
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    predictions.append(response)
    references.append([target])

# 4. Compute Metrics
print("\n📈 Calculating ROUGE and BLEU metrics for BASE MODEL...")
try:
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    print("\n" + "="*50)
    print("🎯 FINAL EVALUATION METRICS (BASE MODEL ONLY) 🎯")
    print("="*50)
    print(f"SacreBLEU Score: {bleu_score['score']:.2f}")
    print(f"ROUGE-1:         {rouge_score['rouge1'] * 100:.2f}")
    print(f"ROUGE-2:         {rouge_score['rouge2'] * 100:.2f}")
    print(f"ROUGE-L:         {rouge_score['rougeL'] * 100:.2f}")
    print("="*50)

except ImportError:
    print("⚠️ Please run: pip install sacrebleu rouge-score evaluate")
