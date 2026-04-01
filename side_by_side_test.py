
import os
import sys
import torch
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIG ---
BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/qwen_gu_health_lora"
OLLAMA_BASE_ENDPOINT = "http://localhost:11434/api/chat"

def get_ollama_base_response(query):
    """Gets response from the BASE model currently in Ollama."""
    try:
        # Use 'qwen2.5:3b' or similar base model tag in Ollama
        response = requests.post(OLLAMA_BASE_ENDPOINT, json={
            "model": "qwen2.5:3b",
            "messages": [
                {"role": "system", "content": "You are a professional Gujarati Healthcare Assistant."},
                {"role": "user", "content": query}
            ],
            "stream": False
        })
        return response.json().get("message", {}).get("content", "Error fetching Ollama response.")
    except Exception as e:
        return f"Ollama Error: {e}"

def main():
    query = "ડાયાબિટીઝના મુખ્ય લક્ષણો અને તેના ઉપાય કયા છે?" # Symptoms and remedies of diabetes
    print(f"\n🔍 TESTING QUERY: {query}\n")
    print("="*80)
    
    # 1. Get Base Response (Fast, via Ollama)
    print("⏳ Fetching BASE model response (via Ollama)...")
    base_ans = get_ollama_base_response(query)
    
    # 2. Get Fine-tuned Response (Slow, via Transformers on CPU)
    print("⚙️  Loading FINE-TUNED model (Base + LoRA) to CPU... (Takes ~6GB RAM)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_ID,
            device_map="cpu", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        
        messages = [
            {"role": "system", "content": "You are a safe, accurate Gujarati healthcare assistant."},
            {"role": "user", "content": query}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cpu")
        
        print("🤖  Generating Fine-tuned Answer...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
        
        ft_ans = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        ft_ans = f"Fine-tune Loading Error: {e}"

    # 3. Display Results
    print("\n" + "="*80)
    print("🎯 SIDE-BY-SIDE COMPARISON 🎯")
    print("="*80)
    print(f"\n[1] BASE MODEL (Ollama):\n{'-'*30}\n{base_ans}")
    print(f"\n[2] FINE-TUNED MODEL (Base + LoRA):\n{'-'*30}\n{ft_ans}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
