import random
import string
import sys
import torch # Added for dtype and device management
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Added Hugging Face imports
import gc # Garbage collection
import time # Add time import

print(f"--- Script executed at: {time.time()} ---") # Add this line

# --- Configuration ---
TARGET_CHAR_COUNT = 10_000 # ~2.5k tokens # Start smaller (e.g., 1M chars ~ 250k tokens), increase later to ~40M chars for 10M tokens
NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumQuasar'."
QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"

# --- Hugging Face Model Configuration ---
# !!! IMPORTANT: Replace with the actual model ID when released !!!
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Official ID from Hugging Face

# Configuration for 4-bit quantization (requires bitsandbytes)
USE_4BIT_QUANTIZATION = True # Set to False if not using quantization or hardware doesn't support it well
# --- ---

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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config if USE_4BIT_QUANTIZATION else None,
            torch_dtype=torch.bfloat16, # Use bfloat16 for faster computation / less memory
            device_map="auto", # Requires accelerate - distribute model across available GPUs/CPU/RAM
            max_memory={0: "75GiB"}, # Limit memory only for GPU 0
            trust_remote_code=True,
            # attn_implementation="flash_attention_2" # Optional: Requires flash-attn library, might speed up attention
        )
        print("Model loaded.")

        # --- Tokenize Prompt and Run Inference ---
        print("\n--- Tokenizing Prompt ---")
        # Ensure prompt fits tokenizer context length if applicable (though Scout aims for 10M)
        inputs = tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=False).to(model.device)
        print(f"Input token count: {inputs['input_ids'].shape[1]}")


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
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up GPU memory
        del model
        del tokenizer
        del inputs
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n--- Cleanup Complete ---")


    # 4. Placeholder for LLM Interaction (Old section removed, replaced by above)
    # print("\n--- Prompt Ready ---")
    # print(f"Prompt length (approx characters): {len(final_prompt):,}")
    # print(final_prompt) # Uncomment carefully - this will be HUGE!

    print("\n--- Next Steps (If running locally/on instance) ---")
    # print("1. Copy the generated prompt (or save it to a file).") # No longer needed if running here
    # print("2. Send this prompt to the Llama 4 Scout model endpoint/API.") # Now handled in script
    print("1. Ensure you have replaced the placeholder MODEL_ID with the correct one.")
    print("2. Ensure this script runs on a machine with the required hardware (H100 GPU).")
    print("3. Install all dependencies from requirements.txt.")
    print(f"4. Adjust TARGET_CHAR_COUNT (currently {TARGET_CHAR_COUNT:,}) to target ~10M tokens (~40M chars).")
    print("5. Run the script: python long_context_test.py")
    print(f"6. Evaluate the model's response printed above.")


    # Example of how you might save the prompt to a file (Optional now)
    # try:
    #     output_filename = "long_prompt.txt"
    #     with open(output_filename, "w", encoding="utf-8") as f:
    #         f.write(final_prompt)
    #     print(f"\nPrompt saved to '{output_filename}'. Size: {len(final_prompt) / (1024*1024):.2f} MB")
    # except Exception as e:
    #     print(f"\nError saving prompt to file: {e}")
    #     print("You might need to handle the large prompt string directly.")


if __name__ == "__main__":
    # Increase recursion depth for potentially deep structures if needed, though less likely here
    # sys.setrecursionlimit(20000)
    main() 