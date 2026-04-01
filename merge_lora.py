
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "models/qwen_gu_health_lora"
MERGED_SAVE_PATH = "models/qwen_gu_health_merged"

def merge_and_save():
    print(f"🚀  Loading Base Model: {BASE_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    
    # Load on CPU to avoid VRAM issues during merge
    print("⚙️  Loading full model (FP16) on CPU for merging...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print(f"⚡  Attaching LoRA weights from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    # Actually 'merge' weights into the core layers
    print("🧬  Merging LoRA weights into Base Model... (This takes a moment)")
    merged_model = model.merge_and_unload()
    
    # Save the consolidated model
    print(f"💾  Saving Merged Model to: {MERGED_SAVE_PATH}")
    merged_model.save_pretrained(MERGED_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_SAVE_PATH)
    
    print("\n✅  MERGE COMPLETE! You now have a unified PyTorch model.")
    print(f"Next Step: Point your pipeline to {MERGED_SAVE_PATH} instead of Ollama for true fine-tuned power.")

if __name__ == "__main__":
    if os.path.exists(LORA_PATH):
        merge_and_save()
    else:
        print(f"❌  Error: LoRA path not found at {LORA_PATH}")
