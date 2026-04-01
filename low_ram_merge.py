
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from tqdm import tqdm

# --- CONFIG ---
BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "models/qwen_gu_health_lora"
SAVE_PATH = "models/qwen_gu_health_merged"

def low_ram_merge():
    print(f"🚀 Starting LOW-RAM Merge for {BASE_ID}")
    
    # 1. Load Tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH if os.path.exists(LORA_PATH) else BASE_ID)
    
    # 2. Load Model in float16 but with DEVICE_MAP='cpu' 
    # and LOW_CPU_MEM_USAGE to prevent spikes
    print("⚙️  Loading model in chunks (Floating Point 16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_ID,
            torch_dtype=torch.float16,
            device_map="cpu", # Force entirely on CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print(f"⚡  Loading LoRA adapter from {LORA_PATH}...")
        model = PeftModel.from_pretrained(
            model, 
            LORA_PATH,
            device_map="cpu",
            torch_dtype=torch.float16
        )
        
        # 3. Perform the actual merge
        print("🧬  Merging LoRA weights (Layer-by-Layer)... This stays efficient.")
        model = model.merge_and_unload()
        
        # 4. Save the result
        print(f"💾  Saving the final merged model to {SAVE_PATH}...")
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH, safe_serialization=True)
        tokenizer.save_pretrained(SAVE_PATH)
        
        print("\n✅ SUCCESS! The SMART model is now saved in 'models/qwen_gu_health_merged'.")
        print("You can now restart your DBs and Ollama.")

    except Exception as e:
        print(f"❌ Merge Failed: {e}")
        print("\nTIP: If it still crashes, try closing your web browser or any other heavy apps temporarily.")

if __name__ == "__main__":
    low_ram_merge()
