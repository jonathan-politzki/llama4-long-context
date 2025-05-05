import random
import string
import sys
import torch # Added for dtype and device management
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Added Hugging Face imports
import gc # Garbage collection
import time # Add time import
import os # For environment variables
import psutil # For memory monitoring
import json

print(f"--- iRoPE Long Context Test (400K tokens) executed at: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

# --- Configuration ---
# Target 400K tokens = ~1.6M characters (assuming average 4 chars per token)
TARGET_CHAR_COUNT = 1600000
NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumQuasar'."
QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"

# --- Hugging Face Model Configuration ---
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Configuration for memory management - these will be checked at runtime
# and only used if available
USE_4BIT_QUANTIZATION = True  # Will be auto-disabled if bitsandbytes not available
ENABLE_CPU_OFFLOAD = True
MAX_GPU_MEMORY = "60GiB"
MAX_CPU_MEMORY = "200GiB"
BATCH_SIZE = 1
USE_FLASH_ATTENTION = False  # Changed to False by default since it's causing issues
ENABLE_CHECKPOINT_ACTIVATION = True

# Add option to track statistics
LOG_RESULTS = True
RESULTS_FILE = "irope_400k_test_results.json"

# --- Environment settings ---
# CUDA memory management to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# Enable iRoPE specifically
os.environ["LLAMA_USE_IROPE"] = "1"  # Signal to use iRoPE in implementation

# --- Runtime flags that will be set based on available libraries ---
_has_bitsandbytes = False
_has_flash_attention = False

# Check for library availability
def check_library_availability():
    global _has_bitsandbytes, _has_flash_attention, USE_4BIT_QUANTIZATION, USE_FLASH_ATTENTION
    
    print("Checking for optional library availability...")
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        _has_bitsandbytes = True
        print("✅ bitsandbytes is available - 4-bit quantization enabled")
    except (ImportError, ModuleNotFoundError):
        _has_bitsandbytes = False
        USE_4BIT_QUANTIZATION = False
        print("❌ bitsandbytes is not available - 4-bit quantization disabled")
    
    # Check for flash-attention
    if USE_FLASH_ATTENTION:
        try:
            import flash_attn
            _has_flash_attention = True
            print("✅ flash-attention is available")
        except (ImportError, ModuleNotFoundError):
            _has_flash_attention = False
            USE_FLASH_ATTENTION = False
            print("❌ flash-attention is not available - using standard attention mechanism")
    else:
        print("ℹ️ flash-attention is disabled in configuration")

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
    """Generates more realistic filler text to simulate a long document."""
    print(f"Generating approximately {char_count:,} characters of filler text...")
    
    paragraphs = []
    current_length = 0
    
    # Generate a more varied content with paragraph structures
    while current_length < char_count:
        # Create varied paragraph lengths
        para_length = random.randint(200, 800)
        sentences = []
        
        # Generate sentences for this paragraph
        while sum(len(s) for s in sentences) < para_length and current_length < char_count:
            # Create varied sentence templates
            templates = [
                "The analysis of {} indicates that {} is a critical factor in {}.",
                "Research conducted by {} demonstrates significant implications for {} in the context of {}.",
                "According to {}, the relationship between {} and {} requires further investigation.",
                "The implementation of {} has resulted in unexpected consequences for {} and {}.",
                "Despite concerns about {}, evidence suggests that {} remains effective for {}.",
                "Experts in {} continue to debate whether {} actually influences {}.",
                "The development of {} represents a major advancement in how we understand {} and {}.",
                "Historical data regarding {} contradicts current theories about {} in relation to {}.",
                "The impact of {} on {} has been significantly overestimated according to recent {} studies.",
                "Preliminary findings regarding {} suggest a correlation with {} under specific {} conditions."
            ]
            
            # Fill in templates with random terms
            terms = [
                "machine learning", "neural networks", "computer vision", "natural language processing",
                "quantum computing", "blockchain technology", "cloud infrastructure", "data privacy",
                "artificial intelligence", "systems architecture", "ethical considerations", "user experience",
                "algorithmic bias", "computational efficiency", "resource allocation", "security protocols",
                "network topology", "distributed systems", "parameter optimization", "data structures",
                "memory management", "compiler design", "operating systems", "virtualization",
                "database management", "information retrieval", "knowledge graphs", "semantic analysis"
            ]
            
            template = random.choice(templates)
            filled_template = template.format(
                random.choice(terms),
                random.choice(terms),
                random.choice(terms)
            )
            
            sentences.append(filled_template)
            current_length += len(filled_template) + 1  # +1 for space
        
        # Join sentences into a paragraph
        paragraph = " ".join(sentences)
        paragraphs.append(paragraph)
        
        # Early exit if we've exceeded the target
        if current_length >= char_count:
            break
    
    # Join paragraphs with double newlines
    text = "\n\n".join(paragraphs)
    
    # Trim to exact length if needed
    if len(text) > char_count:
        text = text[:char_count]
    
    print(f"Generated {len(text):,} characters of filler text.")
    return text

def insert_needle(haystack, needle, position_percentage=None):
    """
    Inserts the needle into a specific position within the haystack.
    
    Args:
        haystack: The large text to insert the needle into
        needle: The text to insert
        position_percentage: Optional, if provided will insert at approximately 
                            this percentage through the text (0-100)
    """
    print("Inserting needle into haystack...")
    
    # Determine insertion point
    if position_percentage is not None:
        # Convert percentage to position
        percentage = max(0, min(100, position_percentage)) / 100
        insertion_point = int(len(haystack) * percentage)
    else:
        # Random insertion between 10% and 90% of the document
        max_insertion_point = max(0, len(haystack) - len(needle) - 100)
        min_point = int(len(haystack) * 0.1)
        max_point = int(len(haystack) * 0.9)
        insertion_point = random.randint(min_point, max_point) if max_insertion_point > 100 else len(haystack) // 2

    # Calculate percentage position for reporting
    actual_percentage = (insertion_point / len(haystack)) * 100
    
    haystack_with_needle = haystack[:insertion_point] + "\n\n--- NEEDLE START --- \n" + needle + "\n--- NEEDLE END --- \n\n" + haystack[insertion_point:]
    print(f"Needle inserted at character position {insertion_point:,} ({actual_percentage:.1f}% through the document).")
    return haystack_with_needle, actual_percentage

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
    
    # Ensure we have the right libraries available
    check_library_availability()
    
    # Configure quantization if available
    quantization_config = None
    if USE_4BIT_QUANTIZATION and _has_bitsandbytes:
        print("Configuring 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Create explicit device map for offloading
    device_map = "auto"  # Default to auto
    offload_folder = None
    
    if ENABLE_CPU_OFFLOAD:
        print("Configuring CPU offloading...")
        offload_folder = "./offload_folder"
        # Create folder if needed
        if not os.path.exists(offload_folder):
            os.makedirs(offload_folder)
            
        # Create more specific device map if explicitly offloading
        device_map = {
            "model.embed_tokens": 0,  # Keep embeddings on GPU
            "model.norm": 0,          # Keep final normalization on GPU
            "lm_head": 0,             # Keep language model head on GPU
        }
    
    # Configure model loading options
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "max_memory": {0: MAX_GPU_MEMORY, "cpu": MAX_CPU_MEMORY},
        "trust_remote_code": True,
    }
    
    # Add quantization config if available
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    
    # Add offloading options if enabled
    if ENABLE_CPU_OFFLOAD:
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_state_dict"] = True
    
    # Add flash attention if available and enabled
    if USE_FLASH_ATTENTION and _has_flash_attention:
        print("Enabling Flash Attention 2...")
        kwargs["attn_implementation"] = "flash_attention_2"
    
    # Print key configuration
    print(f"Model loading configuration:")
    print(f"- 4-bit quantization: {'Enabled' if USE_4BIT_QUANTIZATION and _has_bitsandbytes else 'Disabled'}")
    print(f"- CPU offloading: {'Enabled' if ENABLE_CPU_OFFLOAD else 'Disabled'}")
    print(f"- Flash Attention: {'Enabled' if USE_FLASH_ATTENTION and _has_flash_attention else 'Disabled'}")
    
    try:
        # Load model with configured options
        print(f"Loading model {MODEL_ID}...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        
        # Apply model-specific optimizations
        if hasattr(model, "config") and hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1  # Ensure tensor parallelism is disabled
        
        # Enable the use of iRoPE if supported by the model
        if hasattr(model.config, "rope_theta"):
            print("Setting RoPE theta scaling for better length extrapolation...")
            # Use a lower base frequency for better long-context handling
            model.config.rope_theta = 10000.0 * 2.0
            print(f"RoPE theta set to: {model.config.rope_theta}")
        
        if hasattr(model.config, "rope_scaling"):
            print("Configuring RoPE scaling parameters for length extrapolation...")
            # Apply scaling factor for RoPE to better handle longer sequences
            model.config.rope_scaling = {"type": "linear", "factor": 4.0}
            print(f"RoPE scaling config: {model.config.rope_scaling}")
        
        # Log if iRoPE is being used (depends on model implementation)
        if os.environ.get("LLAMA_USE_IROPE") == "1":
            print("iRoPE is explicitly enabled via environment variable")
        
        if ENABLE_CHECKPOINT_ACTIVATION:
            model.gradient_checkpointing_enable()
            
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        
        if "bitsandbytes" in str(e).lower() and USE_4BIT_QUANTIZATION:
            print("\nTrying again without 4-bit quantization...")
            # Retry without quantization
            USE_4BIT_QUANTIZATION = False
            return load_model_with_extreme_memory_savings()
            
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("\nAuthentication error with Hugging Face!")
            print("Please make sure you have access to this model and have logged in with:")
            print("  huggingface-cli login")
            print("Or set the HF_TOKEN environment variable with your access token.")
            
        # Re-raise the exception to be caught by the main function
        raise

def process_with_streaming(tokenizer, model, prompt):
    """Process the prompt and generate response with streaming output."""
    print("\nTokenizing input...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    total_tokens = input_ids.shape[1]
    print(f"Input has {total_tokens:,} tokens")
    
    print("\nGenerating response...")
    start_time = time.time()
    
    # Configure generation parameters
    generation_config = {
        "max_new_tokens": 150,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_beams": 1,  # Disable beam search for efficiency with long contexts
        "repetition_penalty": 1.1,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    
    # Stream tokens as they're generated
    response_text = ""
    
    # Generate with streamed output
    print("\nModel response:")
    print("=" * 40)
    
    try:
        streamer = None  # Initialize to suppress potential warning
        
        # Generate output tokens
        outputs = model.generate(
            input_ids,
            streamer=streamer,
            **generation_config
        )
        
        # Get the generated text
        output_text = tokenizer.decode(outputs.sequences[0, total_tokens:], skip_special_tokens=True)
        print(output_text)
        response_text = output_text
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        response_text = f"[Generation Error: {str(e)}]"
    
    print("=" * 40)
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    
    return response_text, generation_time, total_tokens

def evaluate_response(response_text, needle):
    """Evaluate if the response correctly extracted the needle information."""
    # Check for the key information in the needle
    key_info = "QuantumQuasar"
    found = key_info.lower() in response_text.lower()
    
    print("\n--- Evaluation ---")
    print(f"Needle successfully retrieved: {'✅ YES' if found else '❌ NO'}")
    print(f"Looking for: '{key_info}'")
    if not found:
        print("The model failed to retrieve the needle information.")
    return found

def log_results(results_dict):
    """Log test results to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_FILE) if os.path.dirname(RESULTS_FILE) else '.', exist_ok=True)
    
    # Check if file exists to append or create new
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    existing_results.append(results_dict)
                else:
                    existing_results = [existing_results, results_dict]
            except json.JSONDecodeError:
                existing_results = [results_dict]
    else:
        existing_results = [results_dict]
    
    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"Results logged to {RESULTS_FILE}")

def main():
    print("--- Starting iRoPE 400K Token Context Test ---")
    print_system_info()

    # Check available libraries before proceeding
    check_library_availability()

    # Track test statistics
    test_stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_char_count": TARGET_CHAR_COUNT,
        "target_tokens": TARGET_CHAR_COUNT // 4,  # Estimate
        "model_id": MODEL_ID,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "total_gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
            "total_ram_gb": psutil.virtual_memory().total / (1024**3)
        },
        "settings": {
            "4bit_quantization": USE_4BIT_QUANTIZATION,
            "flash_attention": USE_FLASH_ATTENTION,
            "cpu_offload": ENABLE_CPU_OFFLOAD
        }
    }

    # 1. Generate Haystack
    print("\n--- Step 1: Generating haystack text ---")
    try:
        haystack = generate_filler_text(TARGET_CHAR_COUNT)
        test_stats["actual_char_count"] = len(haystack)
    except Exception as e:
        print(f"❌ Error generating haystack: {e}")
        if LOG_RESULTS:
            test_stats["status"] = "error"
            test_stats["error"] = f"Haystack generation error: {str(e)}"
            log_results(test_stats)
        return

    # 2. Insert Needle at 50% position
    print("\n--- Step 2: Inserting needle into haystack ---")
    try:
        text_with_needle, needle_position = insert_needle(haystack, NEEDLE, position_percentage=50)
        test_stats["needle_position_percent"] = needle_position
    except Exception as e:
        print(f"❌ Error inserting needle: {e}")
        if LOG_RESULTS:
            test_stats["status"] = "error" 
            test_stats["error"] = f"Needle insertion error: {str(e)}"
            log_results(test_stats)
        return

    # 3. Create Prompt
    print("\n--- Step 3: Creating prompt ---")
    try:
        final_prompt = create_prompt(text_with_needle, QUESTION)
        test_stats["prompt_length_chars"] = len(final_prompt)
    except Exception as e:
        print(f"❌ Error creating prompt: {e}")
        if LOG_RESULTS:
            test_stats["status"] = "error"
            test_stats["error"] = f"Prompt creation error: {str(e)}"
            log_results(test_stats)
        return

    # --- Prepare for LLM Interaction ---
    print("\n--- Preparing for Model Interaction ---")
    print(f"Model ID: {MODEL_ID}")
    print(f"Prompt length (approx characters): {len(final_prompt):,}")

    # Clear some memory before loading model
    del haystack
    del text_with_needle
    clear_gpu_memory()

    # --- Load Tokenizer and Model ---
    print("\n--- Loading Tokenizer and Model ---")
    load_start_time = time.time()

    tokenizer = None
    model = None

    try:
        # Load tokenizer
        print(f"Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")

        # Load model with memory optimizations
        print(f"Loading model {MODEL_ID}...")
        model_load_start = time.time()
        model = load_model_with_extreme_memory_savings()
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f} seconds.")
        
        test_stats["model_load_time_seconds"] = model_load_time

        # --- Process prompt and generate response ---
        print("\n--- Processing Long Context and Generating Response ---")
        response_text, generation_time, input_token_count = process_with_streaming(tokenizer, model, final_prompt)
        
        test_stats["input_token_count"] = input_token_count
        test_stats["generation_time_seconds"] = generation_time
        
        # Evaluate response
        success = evaluate_response(response_text, NEEDLE)
        test_stats["needle_found"] = success
        test_stats["response"] = response_text[:1000]  # Save first 1000 chars
        
        # Record peak memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
            test_stats["peak_gpu_memory_gb"] = peak_memory
            print(f"Peak GPU memory usage: {peak_memory:.2f} GB")

        # Log the success or failure
        print("\n--- Test Status ---")
        if success:
            print("✅ SUCCESS: The model successfully retrieved the needle in a 400K context!")
            test_stats["status"] = "success"
        else:
            print("❌ FAILURE: The model failed to retrieve the needle.")
            test_stats["status"] = "failure"

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        test_stats["status"] = "error"
        test_stats["error"] = str(e)
        
        # Check for common errors and provide helpful messages
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("\n--- Authentication Error ---")
            print("You need to log in to Hugging Face to access this model.")
            print("Run: huggingface-cli login")
            print("Or set the HF_TOKEN environment variable.")
        elif "cuda" in error_str or "gpu" in error_str or "memory" in error_str:
            print("\n--- GPU Memory Error ---")
            print("Try reducing TARGET_CHAR_COUNT or disabling features.")
        
    finally:
        # Clean up GPU memory
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        clear_gpu_memory()
        
        # Log the overall test time
        test_stats["total_test_time_seconds"] = time.time() - load_start_time
        print(f"\nTotal test time: {test_stats['total_test_time_seconds']:.2f} seconds")
        
        # Save results to file
        if LOG_RESULTS:
            log_results(test_stats)
        
        print("\n--- Cleanup Complete ---")

    print("\n--- Next Steps ---")
    print("1. If this test was successful, try increasing context size further")
    print("2. If it failed, try reducing TARGET_CHAR_COUNT or enable more optimizations")
    print("3. See the results file for detailed metrics")

if __name__ == "__main__":
    main() 