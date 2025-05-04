import random
import string
import sys

# --- Configuration ---
TARGET_CHAR_COUNT = 1_000_000 # Start smaller (e.g., 1M chars ~ 250k tokens), increase later to ~40M chars for 10M tokens
NEEDLE = "The secret passphrase for the blueberry muffin recipe is 'QuantumQuasar'."
QUESTION = "What is the secret passphrase for the blueberry muffin recipe?"
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

    # 4. Placeholder for LLM Interaction
    print("\n--- Prompt Ready ---")
    print(f"Prompt length (approx characters): {len(final_prompt):,}")
    # print(final_prompt) # Uncomment carefully - this will be HUGE!

    print("\n--- Next Steps ---")
    print("1. Copy the generated prompt (or save it to a file).")
    print("2. Send this prompt to the Llama 4 Scout model endpoint/API.")
    print("3. Evaluate the model's response to see if it contains the 'needle':")
    print(f"   Expected needle substring: '{NEEDLE}'")

    # Example of how you might save the prompt to a file
    try:
        output_filename = "long_prompt.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        print(f"\nPrompt saved to '{output_filename}'. Size: {len(final_prompt) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"\nError saving prompt to file: {e}")
        print("You might need to handle the large prompt string directly.")


if __name__ == "__main__":
    # Increase recursion depth for potentially deep structures if needed, though less likely here
    # sys.setrecursionlimit(20000)
    main() 