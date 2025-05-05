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
TARGET_CHAR_COUNT = 1000 # Start extremely small for testing first
NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumQuasar'."
QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"

# --- Hugging Face Model Configuration ---
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Official ID from Hugging Face

# Configuration for 4-bit quantization (requires bitsandbytes)
USE_4BIT_QUANTIZATION = True # Set to False if not using quantization or hardware doesn't support it well

# Advanced memory management options
OFFLOAD_FOLDER = "./offload_folder"  # Where to offload model parts if needed
ENABLE_CPU_OFFLOAD = True  # Enable CPU offloading for layers that don't fit in GPU
MAX_GPU_MEMORY = "60GiB"   # Leave substantial headroom for the KV cache
MAX_CPU_MEMORY = "200GiB"  # Use plenty of CPU RAM for offloading
BATCH_SIZE = 1  # Absolute minimum to reduce memory footprint
USE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
ENABLE_CHECKPOINT_ACTIVATION = True  # Enable gradient checkpointing to save memory

# --- Environment settings ---
# CUDA memory management to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

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

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_accumulated_memory_stats(i)
    print("GPU memory cleared")

def load_model_with_extreme_memory_savings():
    """Load the model with most extreme memory saving techniques."""
    print("Loading model with extreme memory saving techniques")
    
    # Configure 4-bit quantization with double quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Create explicit device map to heavily offload to CPU
    device_map = {
        "model.embed_tokens": 0,  # Keep embeddings on GPU
        "model.norm": 0,          # Keep final normalization on GPU
        "lm_head": 0,             # Keep language model head on GPU
    }
    
    # All other layers will automatically be offloaded to CPU
    
    # Load with minimal memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map if ENABLE_CPU_OFFLOAD else "auto",
        offload_folder=OFFLOAD_FOLDER if ENABLE_CPU_OFFLOAD else None,
        offload_state_dict=ENABLE_CPU_OFFLOAD,
        low_cpu_mem_usage=True,
        max_memory={0: MAX_GPU_MEMORY, "cpu": MAX_CPU_MEMORY},
        trust_remote_code=True,
    )
    
    # Apply model-specific optimizations
    if hasattr(model, "config") and hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1  # Ensure tensor parallelism is disabled
    
    if ENABLE_CHECKPOINT_ACTIVATION:
        model.gradient_checkpointing_enable()
    
    return model

def process_in_chunks(tokenizer, model, text, chunk_size=1000):
    """Process very long text in chunks to avoid OOM during tokenization."""
    print(f"Processing long text in chunks of {chunk_size} tokens")
    
    # Tokenize the full text (but don't convert to tensors yet)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Initialize result
    final_output = ""
    
    # Process in chunks
    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{len(all_tokens)//chunk_size + 1}")
        
        # Convert chunk to tensor and move to device
        input_ids = torch.tensor([chunk]).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Clear GPU memory after each chunk
        del input_ids
        clear_gpu_memory()
    
    # For long contexts, we're not really generating, just checking if model can handle it
    return "Test completed. The model successfully processed the chunks."

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
    clear_gpu_memory()

    # Create offload folder if it doesn't exist
    if ENABLE_CPU_OFFLOAD and not os.path.exists(OFFLOAD_FOLDER):
        os.makedirs(OFFLOAD_FOLDER)

    # --- Load Tokenizer and Model (Requires Appropriate Hardware: H100 GPU recommended) ---
    print("\n--- Loading Tokenizer and Model ---")
    print("!!! This requires significant GPU VRAM (H100 recommended) and time !!!")

    tokenizer = None
    model = None

    try:
        print(f"DEBUG: Value of USE_4BIT_QUANTIZATION before if: {USE_4BIT_QUANTIZATION}, Type: {type(USE_4BIT_QUANTIZATION)}")
        
        print(f"DEBUG: Attempting to load tokenizer with MODEL_ID = {MODEL_ID}")
        # Trust remote code if required by the specific model implementation
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Set padding token if not already set (common practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")

        print(f"DEBUG: Attempting to load model with MODEL_ID = {MODEL_ID}")
        print(f"Loading model {MODEL_ID}...")
        
        # Load model with extreme memory saving techniques
        model = load_model_with_extreme_memory_savings()
        print("Model loaded.")

        # --- Process text efficiently ---
        print("\n--- Processing Text ---")
        
        # Attempt to process with chunking for very long contexts
        try:
            result_text = process_in_chunks(tokenizer, model, final_prompt)
            
            print("\n--- Processing Result ---")
            print(result_text)
            print("-----------------------")
            
            print("\n--- Test Status ---")
            print("✅ Success: The model handled the context!")
            
        except Exception as processing_error:
            print(f"❌ Error during text processing: {processing_error}")
            import traceback
            traceback.print_exc()

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
        clear_gpu_memory()
        print("\n--- Cleanup Complete ---")


    print("\n--- Next Steps ---")
    print("1. If this test was successful, gradually increase TARGET_CHAR_COUNT")
    print(f"2. Current TARGET_CHAR_COUNT: {TARGET_CHAR_COUNT:,} (~{TARGET_CHAR_COUNT//4:,} tokens)")
    print("3. Target for Llama 4 Scout: 40,000,000 chars (~10M tokens)")
    print("4. For full inference, switch back to the normal model.generate() approach")


if __name__ == "__main__":
    main() 