#!/usr/bin/env python3
"""
Long Context Comparison: Llama 4 Scout vs Gemini
------------------------------------------------
This script performs a side-by-side comparison of Llama 4 Scout and Gemini 
with a 2 million token context window using the needle-in-haystack approach.
"""

import random
import string
import sys
import os
import time
import gc
import torch
import psutil
import json
from pathlib import Path
from datetime import datetime
import argparse

# Hugging Face imports for Llama 4
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai package not found. Gemini API will not be available.")

# --- Configuration ---
# Default values - can be overridden by command-line args
DEFAULT_TARGET_CHAR_COUNT = 8_000_000  # Approx 2M tokens
DEFAULT_NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumFusionParadox42'."
DEFAULT_QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"

# Model Configuration
LLAMA_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Llama 4 Scout model
GEMINI_MODEL = "gemini-1.5-pro"  # Gemini model

# Advanced settings
USE_4BIT_QUANTIZATION = True
OFFLOAD_FOLDER = "./offload_folder"
ENABLE_CPU_OFFLOAD = True
MAX_GPU_MEMORY = "70GiB"
RESULTS_DIR = "./comparison_results"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# --- Environment settings ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare Llama 4 and Gemini on long context tasks")
    parser.add_argument("--char-count", type=int, default=DEFAULT_TARGET_CHAR_COUNT,
                        help=f"Number of characters for context (default: {DEFAULT_TARGET_CHAR_COUNT})")
    parser.add_argument("--needle", type=str, default=DEFAULT_NEEDLE,
                        help="The needle to hide in the context")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION,
                        help="Question to ask about the needle")
    parser.add_argument("--llama-only", action="store_true", 
                        help="Only run the test on Llama 4")
    parser.add_argument("--gemini-only", action="store_true", 
                        help="Only run the test on Gemini")
    parser.add_argument("--gemini-api-key", type=str, 
                        help="Gemini API key (can also be set via GEMINI_API_KEY env var)")
    return parser.parse_args()

def print_system_info():
    """Print system information for debugging purposes."""
    print("\n--- System Information ---")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    else:
        print("CUDA Available: No")
    
    vm = psutil.virtual_memory()
    print(f"Total System RAM: {vm.total / (1024**3):.2f} GB")
    print(f"Available System RAM: {vm.available / (1024**3):.2f} GB")
    print(f"CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print("---------------------------")

def generate_filler_text(char_count):
    """Generates repetitive filler text to approximate the haystack size."""
    print(f"Generating approximately {char_count:,} characters of filler text...")
    # Simple repetition - more complex generation can be added later
    base_phrase = "This is filler text to simulate a large context window. "
    repetitions = (char_count // len(base_phrase)) + 1
    text = (base_phrase * repetitions)[:char_count]
    print(f"Filler text generated: {len(text):,} characters")
    return text

def insert_needle(haystack, needle):
    """Inserts the needle into a random position within the haystack."""
    print("Inserting needle into haystack...")
    # Ensure enough space before/after for clear insertion if possible
    max_insertion_point = max(0, len(haystack) - len(needle) - 100)
    insertion_point = random.randint(100, max_insertion_point) if max_insertion_point > 100 else len(haystack) // 2

    haystack_with_needle = haystack[:insertion_point] + "\n\n--- NEEDLE START --- \n" + needle + "\n--- NEEDLE END --- \n\n" + haystack[insertion_point:]
    print(f"Needle inserted near character position {insertion_point:,}.")
    return haystack_with_needle, insertion_point

def create_prompt(text_with_needle, question):
    """Formats the final prompt for the LLM."""
    prompt = f"""Here is a very long document:

<document>
{text_with_needle}
</document>

Based *only* on the document provided above, please answer the following question:
Question: {question}
Answer:"""
    return prompt

def test_llama(prompt, needle, args):
    """Run the test on Llama 4 Scout."""
    print("\n=== Testing Llama 4 Scout ===")
    
    # Create offload folder if it doesn't exist
    if ENABLE_CPU_OFFLOAD and not os.path.exists(OFFLOAD_FOLDER):
        os.makedirs(OFFLOAD_FOLDER)
    
    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        start_time = time.time()
        
        # Configure quantization if enabled
        quantization_config = None
        if USE_4BIT_QUANTIZATION:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        print(f"Loading tokenizer for {LLAMA_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with advanced memory management
        print(f"Loading model {LLAMA_MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID,
            quantization_config=quantization_config if USE_4BIT_QUANTIZATION else None,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER if ENABLE_CPU_OFFLOAD else None,
            offload_state_dict=ENABLE_CPU_OFFLOAD,
            low_cpu_mem_usage=True,
            max_memory={0: MAX_GPU_MEMORY, "cpu": "64GiB"},
            trust_remote_code=True,
        )
        
        # Tokenize input
        print("Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False).to("cuda:0")
        print(f"Input token count: {inputs['input_ids'].shape[1]:,}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Usage after tokenization: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Generate response
        print("Running inference...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode response
        response_tokens = outputs[0, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Evaluate
        success = needle.lower() in response.lower()
        
        result = {
            "model": "Llama 4 Scout",
            "model_id": LLAMA_MODEL_ID,
            "success": success,
            "runtime_seconds": duration,
            "token_count": inputs['input_ids'].shape[1],
            "response": response,
        }
        
        # Print results
        print(f"\n--- Llama 4 Scout Response ---")
        print(response)
        print("-----------------------")
        print(f"Success: {'✅' if success else '❌'}")
        print(f"Runtime: {duration:.2f} seconds")
        
        return result
    
    except Exception as e:
        print(f"❌ Error testing Llama 4: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": "Llama 4 Scout",
            "model_id": LLAMA_MODEL_ID,
            "error": str(e),
            "success": False,
        }
    
    finally:
        # Cleanup
        if 'model' in locals() and 'model' in vars():
            del model
        if 'tokenizer' in locals() and 'tokenizer' in vars():
            del tokenizer
        if 'inputs' in locals() and 'inputs' in vars():
            del inputs
        if 'outputs' in locals() and 'outputs' in vars():
            del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_gemini(prompt, needle, args):
    """Run the test on Gemini."""
    print("\n=== Testing Gemini ===")
    
    if not GEMINI_AVAILABLE:
        return {
            "model": "Gemini",
            "model_id": GEMINI_MODEL,
            "error": "google-generativeai package not installed",
            "success": False,
        }
    
    # Get API key from args or environment
    api_key = args.gemini_api_key or os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        return {
            "model": "Gemini",
            "model_id": GEMINI_MODEL,
            "error": f"No Gemini API key provided. Set with --gemini-api-key or {GEMINI_API_KEY_ENV} env var",
            "success": False,
        }
    
    try:
        start_time = time.time()
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model instance
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Count approximate tokens
        # 1 token ≈ 4 characters in English
        approx_tokens = len(prompt) // 4
        print(f"Approximate token count: {approx_tokens:,}")
        
        # Generate response
        print("Sending request to Gemini API...")
        response = model.generate_content(prompt)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract response text
        response_text = response.text
        
        # Evaluate
        success = needle.lower() in response_text.lower()
        
        result = {
            "model": "Gemini",
            "model_id": GEMINI_MODEL,
            "success": success,
            "runtime_seconds": duration,
            "approx_token_count": approx_tokens,
            "response": response_text,
        }
        
        # Print results
        print(f"\n--- Gemini Response ---")
        print(response_text)
        print("-----------------------")
        print(f"Success: {'✅' if success else '❌'}")
        print(f"Runtime: {duration:.2f} seconds")
        
        return result
    
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": "Gemini",
            "model_id": GEMINI_MODEL,
            "error": str(e),
            "success": False,
        }

def save_results(results, prompt, args):
    """Save test results to a JSON file."""
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Assemble results object with metadata
    full_results = {
        "timestamp": timestamp,
        "config": {
            "char_count": args.char_count,
            "needle": args.needle,
            "question": args.question,
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None,
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
        },
        "prompt_length": len(prompt),
        "prompt_sample": prompt[:500] + "..." + prompt[-500:] if len(prompt) > 1000 else prompt,
        "results": results
    }
    
    # Save to file
    output_file = os.path.join(RESULTS_DIR, f"comparison_results_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def main():
    """Main function to run the comparison."""
    args = parse_args()
    print("=== Long Context Model Comparison ===")
    print(f"Target context size: {args.char_count:,} characters (~{args.char_count//4:,} tokens)")
    print_system_info()
    
    # Set up test data
    haystack = generate_filler_text(args.char_count)
    text_with_needle, insertion_point = insert_needle(haystack, args.needle)
    prompt = create_prompt(text_with_needle, args.question)
    
    # Print prompt stats
    print(f"\nPrompt length: {len(prompt):,} characters (~{len(prompt)//4:,} tokens)")
    print(f"Needle position: {insertion_point:,} characters from start")
    
    # Free memory after creating prompt
    del haystack
    gc.collect()
    
    # Results collection
    results = []
    
    # Run tests based on flags
    if args.gemini_only:
        gemini_result = test_gemini(prompt, args.needle, args)
        results.append(gemini_result)
    elif args.llama_only:
        llama_result = test_llama(prompt, args.needle, args)
        results.append(llama_result)
    else:
        # Run both tests
        llama_result = test_llama(prompt, args.needle, args)
        results.append(llama_result)
        
        gemini_result = test_gemini(prompt, args.needle, args)
        results.append(gemini_result)
    
    # Save results
    save_results(results, prompt, args)
    
    # Print summary
    print("\n=== Comparison Summary ===")
    for result in results:
        model_name = result.get("model")
        success = result.get("success", False)
        error = result.get("error", None)
        
        if error:
            print(f"{model_name}: ❌ Error - {error}")
        else:
            runtime = result.get("runtime_seconds", float('nan'))
            print(f"{model_name}: {'✅' if success else '❌'} ({runtime:.2f}s)")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main() 