"""
Gemini API integration for long context testing.
"""
import os
import time
import google.generativeai as genai
import sys
import json

# Add parent directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.text_generation import generate_filler_text, insert_needle, create_prompt
from utils.metrics import evaluate_response, log_results, get_timestamp

# Gemini API settings
GEMINI_MODEL = "models/gemini-1.5-pro-latest"  # Using the latest Gemini 1.5 Pro model
API_KEY_ENV_VAR = "GEMINI_API_KEY"

def setup_gemini_api():
    """
    Set up the Gemini API with the API key.
    Returns True if successful, False otherwise.
    """
    # Check for API key in environment
    api_key = os.environ.get(API_KEY_ENV_VAR)
    
    if not api_key:
        print(f"❌ Error: No Gemini API key found in environment variable {API_KEY_ENV_VAR}")
        print(f"Please set the API key with: export {API_KEY_ENV_VAR}=your-api-key")
        return False
    
    # Configure the Gemini API
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        return False

def get_model_info():
    """Get information about available Gemini models."""
    try:
        models = genai.list_models()
        gemini_models = [model for model in models if "gemini" in model.name.lower()]
        return gemini_models
    except Exception as e:
        print(f"❌ Error listing Gemini models: {e}")
        return []

def run_gemini_test(char_count, needle, question, needle_key=None, position_percentage=50):
    """
    Run a needle-in-haystack test with Gemini.
    
    Args:
        char_count: Size of the document in characters
        needle: The text to hide in the document (the "needle")
        question: The question to ask to find the needle
        needle_key: The key piece to look for in the response (defaults to needle if None)
        position_percentage: Where to place the needle (0-100% through document)
    
    Returns:
        Dictionary with test results
    """
    if not setup_gemini_api():
        return {"status": "error", "error": "Failed to set up Gemini API"}
    
    # Track test statistics
    test_stats = {
        "timestamp": get_timestamp(),
        "model": GEMINI_MODEL,
        "target_char_count": char_count,
        "needle": needle,
        "question": question
    }
    
    # Create test content
    try:
        print(f"Generating {char_count:,} characters of text...")
        haystack = generate_filler_text(char_count)
        test_stats["actual_char_count"] = len(haystack)
        
        print(f"Inserting needle at {position_percentage}% position...")
        text_with_needle, actual_position = insert_needle(
            haystack, needle, position_percentage=position_percentage
        )
        test_stats["needle_position_percent"] = actual_position
        
        print("Creating prompt...")
        prompt = create_prompt(text_with_needle, question)
        test_stats["prompt_length_chars"] = len(prompt)
        
        # Estimate tokens (approx 4 chars per token)
        estimated_tokens = len(prompt) // 4
        test_stats["estimated_tokens"] = estimated_tokens
        print(f"Prompt ready: {len(prompt):,} characters (~{estimated_tokens:,} tokens)")
        
        # Call Gemini API
        print("\nCalling Gemini API...")
        start_time = time.time()
        
        # Initialize Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Call the API with the prompt
        response = model.generate_content(prompt)
        
        # Process response
        generation_time = time.time() - start_time
        test_stats["generation_time_seconds"] = generation_time
        
        # Extract the text from the response
        response_text = response.text
        test_stats["response"] = response_text[:1000]  # First 1000 chars
        print(f"\nGemini response (truncated):")
        print("=" * 40)
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("=" * 40)
        
        # Evaluate if the needle was found
        needle_key_to_check = needle_key or (needle.split("'")[1] if "'" in needle else needle)
        success = evaluate_response(response_text, needle_key_to_check)
        test_stats["needle_found"] = success
        test_stats["needle_key"] = needle_key_to_check
        
        # Log success/failure
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}: Gemini {'found' if success else 'did not find'} the needle.")
        test_stats["status"] = "success" if success else "failure"
        
        return test_stats
        
    except Exception as e:
        print(f"❌ Error during Gemini test: {e}")
        import traceback
        traceback.print_exc()
        
        test_stats["status"] = "error"
        test_stats["error"] = str(e)
        return test_stats

def main():
    """Run a demo needle-in-haystack test with Gemini."""
    if not setup_gemini_api():
        return
    
    # Display available models
    print("Available Gemini models:")
    models = get_model_info()
    for model in models:
        print(f"- {model.name}: {model.description}")
    
    # Test parameters
    CHAR_COUNT = 100000  # 100K characters (~25K tokens)
    NEEDLE = "The secret authentication key is 'GeminiLongContext2024'."
    QUESTION = "What is the secret authentication key mentioned in the document?"
    OUTPUT_FILE = "results/gemini_test_results.json"
    
    # Run the test
    print(f"\nRunning Gemini needle-in-haystack test with {CHAR_COUNT:,} characters...")
    result = run_gemini_test(CHAR_COUNT, NEEDLE, QUESTION)
    
    # Log results
    log_results(result, OUTPUT_FILE)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 