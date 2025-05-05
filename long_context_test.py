import random
import string
import sys
import torch # Added for dtype and device management
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Added Hugging Face imports
import gc # Garbage collection
import time # Add time import
import os # For environment variables
import psutil # For memory monitoring

print(f"--- Script executed at: {time.time()} ---") # Add this line

# --- Configuration ---
TARGET_CHAR_COUNT = 10_000 # ~2.5k tokens # Start smaller (e.g., 1M chars ~ 250k tokens), increase later to ~40M chars for 10M tokens
NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumQuasar'."
QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"

# --- Hugging Face Model Configuration ---
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Official ID from Hugging Face

# Configuration for 4-bit quantization (requires bitsandbytes)
USE_4BIT_QUANTIZATION = True # Set to False if not using quantization or hardware doesn't support it well

# Advanced memory management options
OFFLOAD_FOLDER = "./offload_folder"  # Where to offload model parts if needed
ENABLE_CPU_OFFLOAD = True  # Enable CPU offloading for layers that don't fit in GPU
MAX_GPU_MEMORY = "70GiB"   # Leave some headroom for the KV cache

# --- Environment settings ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Help prevent fragmentation

# --- ---

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
    print("Filler text generated.")
    return text

def insert_needle(haystack, needle):
    """Inserts the needle into a random position within the haystack."""
    print("Inserting needle into haystack...")
    # Ensure enough space before/after for clear insertion if possible
    max_insertion_point = max(0, len(haystack) - len(needle) - 100)
    insertion_point = random.randint(100, max_insertion_point) if max_insertion_point > 100 else len(haystack) // 2

    haystack_with_needle = haystack[:insertion_point] + "\n\n--- NEEDLE START --- \n" + needle + "\n--- NEEDLE END --- \n\n" + haystack[insertion_point:]
    print(f"Needle inserted near character position {insertion_point:,}.")
    return haystack_with_needle

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

def main():
    print("--- Starting Long Context Test Prompt Generation ---")
    print_system_info()

    # 1. Generate Haystack
    haystack = generate_filler_text(TARGET_CHAR_COUNT)

    # 2. Insert Needle
    text_with_needle = insert_needle(haystack, NEEDLE)

    # 3. Create Prompt
    final_prompt = create_prompt(text_with_needle, QUESTION)

    # --- Prepare for LLM Interaction ---
    print("\n--- Preparing for Hugging Face Model Interaction ---")
    print(f"Model ID: {MODEL_ID}")
    print(f"Prompt length (approx characters): {len(final_prompt):,}")

    # Clear some memory before loading model if possible
    del haystack
    del text_with_needle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create offload folder if it doesn't exist
    if ENABLE_CPU_OFFLOAD and not os.path.exists(OFFLOAD_FOLDER):
        os.makedirs(OFFLOAD_FOLDER)

    # --- Load Tokenizer and Model (Requires Appropriate Hardware: H100 GPU recommended) ---
    print("\n--- Loading Tokenizer and Model ---")
    print("!!! This requires significant GPU VRAM (H100 recommended) and time !!!")

    tokenizer = None
    model = None
    quantization_config = None

    try:
        print(f"DEBUG: Value of USE_4BIT_QUANTIZATION before if: {USE_4BIT_QUANTIZATION}, Type: {type(USE_4BIT_QUANTIZATION)}")
        if USE_4BIT_QUANTIZATION:
             print("Using 4-bit quantization (requires bitsandbytes library).")
             quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # Recommended type
                bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for compute
                bnb_4bit_use_double_quant=True, # Optional, can improve quality slightly
             )

        print(f"DEBUG: Attempting to load tokenizer with MODEL_ID = {MODEL_ID}")
        # Trust remote code if required by the specific model implementation
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Set padding token if not already set (common practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")

        print(f"DEBUG: Attempting to load model with MODEL_ID = {MODEL_ID}")
        print(f"Loading model {MODEL_ID}...")
        
        # Create a device map specifically offloading some layers to CPU
        from accelerate import infer_auto_device_map
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Try to load the model with more advanced memory management
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config if USE_4BIT_QUANTIZATION else None,
            torch_dtype=torch.bfloat16, # Use bfloat16 for faster computation / less memory
            device_map="auto", # Let accelerate distribute the model
            offload_folder=OFFLOAD_FOLDER if ENABLE_CPU_OFFLOAD else None,
            offload_state_dict=ENABLE_CPU_OFFLOAD, # Offload state dict to CPU if needed
            low_cpu_mem_usage=True,  # More efficient memory usage during loading
            max_memory={0: MAX_GPU_MEMORY, "cpu": "64GiB"}, # Limit GPU memory, allow CPU offloading
            trust_remote_code=True,
        )
        print("Model loaded.")

        # --- Tokenize Prompt and Run Inference ---
        print("\n--- Tokenizing Prompt ---")
        # Ensure prompt fits tokenizer context length if applicable (though Scout aims for 10M)
        inputs = tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=False).to("cuda:0")
        print(f"Input token count: {inputs['input_ids'].shape[1]}")

        # Print memory usage after tokenization
        if torch.cuda.is_available():
            print(f"GPU Memory Usage after tokenization: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")

        print("\n--- Running Inference (This will take a long time!) ---")
        # Generate response
        # Adjust generation parameters as needed (e.g., max_new_tokens, temperature)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150, # Limit output length
            do_sample=False, # Use greedy decoding for factual retrieval
            pad_token_id=tokenizer.pad_token_id # Explicitly set pad token id
        )

        print("\n--- Decoding Response ---")
        # Decode only the newly generated tokens, skipping the input prompt tokens
        response_tokens = outputs[0, inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        print("\n--- Model Response ---")
        print(response_text)
        print("-----------------------")

        # --- Evaluation ---
        print("\n--- Evaluation ---")
        if NEEDLE.lower() in response_text.lower():
             print("✅ Success: Needle found in the response!")
        else:
             print("❌ Failure: Needle NOT found in the response.")
        print(f"   Expected needle substring: '{NEEDLE}'")


    except ImportError as e:
        print(f"\n❌ Error: Required library not found. {e}")
        print("Please ensure 'torch', 'transformers', 'accelerate', and 'bitsandbytes' (for 4-bit) are installed.")
        print("See requirements.txt")
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Error: CUDA Out of Memory!")
        print("The model and/or the KV cache for the 10M context likely exceeded available GPU VRAM.")
        print("Ensure you are running on an H100 GPU with sufficient VRAM and system RAM.")
        print("Try reducing TARGET_CHAR_COUNT or using a machine with more memory.")
        if torch.cuda.is_available():
            print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up GPU memory
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        if 'inputs' in locals() and 'inputs' in vars():
            del inputs
        if 'outputs' in locals() and 'outputs' in vars():
            del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n--- Cleanup Complete ---")


    print("\n--- Next Steps (If running locally/on instance) ---")
    print("1. Ensure you have replaced the placeholder MODEL_ID with the correct one.")
    print("2. Ensure this script runs on a machine with the required hardware (H100 GPU).")
    print("3. Install all dependencies from requirements.txt.")
    print(f"4. Adjust TARGET_CHAR_COUNT (currently {TARGET_CHAR_COUNT:,}) to target ~10M tokens (~40M chars).")
    print("5. Run the script: python long_context_test.py")
    print(f"6. Evaluate the model's response printed above.")


if __name__ == "__main__":
    main() 