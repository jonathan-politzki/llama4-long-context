#!/usr/bin/env python3
"""
Needle-in-haystack test using Gemini API for 2M token contexts.
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the Gemini API module
from models.gemini.gemini_api import run_gemini_test
from utils.metrics import log_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a needle-in-haystack test with Gemini API")
    
    parser.add_argument("--char-count", type=int, default=100000,
                        help="Number of characters in the haystack (default: 100,000)")
    
    parser.add_argument("--needle", type=str, 
                        default="The secret authentication key is 'GeminiLongContext2024'.",
                        help="The needle to hide in the haystack")
    
    parser.add_argument("--question", type=str,
                        default="What is the secret authentication key mentioned in the document?",
                        help="The question to ask to find the needle")
    
    parser.add_argument("--position", type=int, default=50,
                        help="Position of the needle as a percentage (0-100) through the document")
    
    parser.add_argument("--output", type=str, default="results/gemini_test_results.json",
                        help="Path to save the test results JSON")
    
    parser.add_argument("--key", type=str, default=None,
                        help="Specific key to check for in the response (defaults to extracting from needle)")
    
    parser.add_argument("--answer", type=str, default=None,
                        help="Specific answer to look for (e.g., '37 hours')")
    
    # Scaling test parameters
    parser.add_argument("--scaling-test", action="store_true",
                        help="Run a scaling test with multiple sizes")
    
    parser.add_argument("--max-chars", type=int, default=8000000,
                        help="Maximum chars for scaling test (default: 8M)")
    
    return parser.parse_args()

def run_scaling_test(args):
    """Run a test at multiple context sizes to see how performance scales."""
    # Sizes for scaling test
    sizes = [
        100000,      # 100K (~25K tokens)
        400000,      # 400K (~100K tokens)
        1600000,     # 1.6M (~400K tokens)
        4000000,     # 4M (~1M tokens)
        8000000,     # 8M (~2M tokens)
    ]
    
    # Only test sizes up to the max specified
    sizes = [size for size in sizes if size <= args.max_chars]
    
    results = []
    
    for size in sizes:
        print(f"\n{'='*40}")
        print(f"TESTING CONTEXT SIZE: {size:,} characters (~{size//4:,} tokens)")
        print(f"{'='*40}\n")
        
        result = run_gemini_test(
            char_count=size,
            needle=args.needle,
            question=args.question,
            needle_key=args.key,
            position_percentage=args.position,
            answer_key=args.answer
        )
        
        results.append(result)
        
        # Add size-specific output file
        size_output = Path(args.output).parent / f"gemini_test_{size//1000}k.json"
        log_results(result, str(size_output))
    
    # Save combined results
    combined_output = Path(args.output).parent / "gemini_scaling_test_results.json"
    with open(combined_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nScaling test complete! Results saved to {combined_output}")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"{'Size':>10} {'Tokens':>10} {'Success':>10} {'Time (s)':>10}")
    print("-" * 45)
    
    for result in results:
        chars = result.get('target_char_count', 0)
        tokens = result.get('estimated_tokens', chars // 4)
        success = "✅" if result.get('needle_found', False) else "❌"
        time = result.get('generation_time_seconds', 0)
        
        print(f"{chars:>10,} {tokens:>10,} {success:>10} {time:>10.2f}")

def main():
    """Run the Gemini long context test."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.scaling_test:
        run_scaling_test(args)
    else:
        # Run a single test
        print(f"Running Gemini needle-in-haystack test with {args.char_count:,} characters...")
        result = run_gemini_test(
            char_count=args.char_count,
            needle=args.needle,
            question=args.question,
            needle_key=args.key,
            position_percentage=args.position,
            answer_key=args.answer
        )
        
        # Save results
        log_results(result, args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 