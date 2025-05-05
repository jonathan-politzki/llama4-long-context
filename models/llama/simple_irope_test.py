#!/usr/bin/env python3
"""
Simple iRoPE test with a small context window
This helps verify the basic functionality before scaling up to 400K tokens
"""
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time
import os
import sys

print(f"--- Simple iRoPE Test ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")

# --- Configuration ---
# Small context for quick testing
TARGET_CHAR_COUNT = 5000  # ~1250 tokens, very small for testing
NEEDLE = "The secret passphrase is 'TestingIRoPE123'."
QUESTION = "What is the secret passphrase?"

# Model configuration
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
USE_4BIT = False  # Disable quantization for simplicity
USE_FLASH_ATTENTION = False

# Enable iRoPE
os.environ["LLAMA_USE_IROPE"] = "1"

def clear_gpu_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared")

def generate_text(length):
    """Generate simple text with repeating pattern"""
    print(f"Generating {length} characters of text...")
    base = "This is test text for the iRoPE model. "
    return (base * (length // len(base) + 1))[:length]

def insert_needle(text, needle):
    """Insert needle in the middle of text"""
    position = len(text) // 2
    result = text[:position] + f"\n\n--- NEEDLE: {needle} ---\n\n" + text[position:]
    print(f"Needle inserted at position {position}")
    return result

def load_model():
    """Load model with minimal configuration"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"Loading model {MODEL_ID}...")
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            print("Using 4-bit quantization")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("BitsAndBytes not available, continuing without quantization")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        
        # Configure RoPE settings if available
        if hasattr(model.config, "rope_scaling"):
            print("Setting RoPE scaling for long context")
            model.config.rope_scaling = {"type": "linear", "factor": 2.0}
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("\nAuthentication error!")
            print("Please login with: huggingface-cli login")
        sys.exit(1)

def main():
    """Run simple needle in haystack test"""
    # Create test data
    haystack = generate_text(TARGET_CHAR_COUNT)
    text_with_needle = insert_needle(haystack, NEEDLE)
    
    # Format prompt
    prompt = f"""Here is a document:
{text_with_needle}

Based only on the above document, answer this question:
{QUESTION}
"""
    
    print(f"Prompt length: {len(prompt)} characters")
    
    # Load model and tokenizer
    try:
        tokenizer, model = load_model()
        
        # Process input
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_tokens = inputs.input_ids.shape[1]
        print(f"Input token count: {input_tokens}")
        
        # Generate response
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                num_beams=1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
        print("\nModel response:")
        print("-" * 40)
        print(response.strip())
        print("-" * 40)
        
        # Check if needle was found
        success = "TestingIRoPE123" in response
        print(f"\nSuccess: {'✅ YES' if success else '❌ NO'}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        clear_gpu_memory()

if __name__ == "__main__":
    main() 