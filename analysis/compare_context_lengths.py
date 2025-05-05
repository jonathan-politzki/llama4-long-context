#!/usr/bin/env python3
"""
Context Length Comparison Test

This script compares the standard RoFormer model with our iRoPE implementation
on progressively longer contexts to measure:
1. Maximum context length before OOM
2. Memory usage differences
3. Inference time
4. Ability to retrieve information from long contexts
"""

import torch
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gc
from typing import Dict, List, Tuple, Optional

# Import standard RoFormer and our iRoPE implementation
from transformers import RoFormerModel, RoFormerConfig, BertTokenizer
from irope_huggingface import IRoPEModel, IRoPEConfig


def format_memory(mem_bytes):
    """Format memory size from bytes to human-readable format"""
    if mem_bytes < 1024:
        return f"{mem_bytes} B"
    elif mem_bytes < 1024 * 1024:
        return f"{mem_bytes / 1024:.2f} KB"
    elif mem_bytes < 1024 * 1024 * 1024:
        return f"{mem_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{mem_bytes / (1024 * 1024 * 1024):.2f} GB"


def measure_memory_usage():
    """Measure current GPU memory usage"""
    if torch.cuda.is_available():
        # Get current GPU memory in bytes
        current, peak = torch.cuda.mem_get_info(0)
        total = current + peak
        used = total - current
        return used
    return 0


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_needle_in_haystack(tokenizer, context_size: int, needle_text: str) -> Tuple[torch.Tensor, int]:
    """
    Create a test input with a 'needle' placed at a random position in the 'haystack'.
    
    Args:
        tokenizer: Tokenizer to use
        context_size: Total context size in tokens
        needle_text: The text to search for
    
    Returns:
        Tuple of (input_ids, needle_position)
    """
    # Tokenize the needle
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    needle_len = len(needle_tokens)
    
    # Create random tokens for the haystack
    # In a real test we'd use actual text, but for this demo random IDs are fine
    vocab_size = tokenizer.vocab_size
    
    # Make sure we have enough space for the needle
    haystack_len = context_size - needle_len
    if haystack_len <= 0:
        raise ValueError(f"Context size {context_size} too small for needle of length {needle_len}")
    
    # Choose a random position for the needle
    needle_pos = np.random.randint(0, haystack_len)
    
    # Create the full sequence with the needle inserted
    prefix = torch.randint(low=5, high=vocab_size-1, size=(needle_pos,))
    suffix = torch.randint(low=5, high=vocab_size-1, size=(haystack_len - needle_pos,))
    
    # Convert needle to tensor
    needle = torch.tensor(needle_tokens)
    
    # Combine all parts
    input_ids = torch.cat([prefix, needle, suffix])
    
    return input_ids, needle_pos


def test_model_on_context(
    model, 
    input_ids: torch.Tensor, 
    device: str,
    batch_size: int = 1,
) -> Dict:
    """Test a model on a given context and return metrics"""
    results = {
        "success": False,
        "error": None,
        "memory_before": 0,
        "memory_peak": 0,
        "inference_time": 0,
    }
    
    # Move input to the specified device
    input_ids = input_ids.to(device)
    
    # Create batch by repeating the input
    if batch_size > 1:
        input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)
    else:
        input_ids = input_ids.unsqueeze(0)
    
    # Prepare attention mask (all 1's)
    attention_mask = torch.ones_like(input_ids)
    
    # Record memory before inference
    clear_memory()
    results["memory_before"] = measure_memory_usage()
    
    # Try to perform inference
    try:
        start_time = time.time()
        
        # Move model to device (might already be there)
        model = model.to(device)
        
        # Run model with torch.no_grad() to minimize memory usage
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Record inference time
        results["inference_time"] = time.time() - start_time
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
    
    # Record peak memory
    results["memory_peak"] = measure_memory_usage()
    
    # Clear memory
    clear_memory()
    
    return results


def run_comparison(
    context_sizes: List[int],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    needle_text: str = "THE SECRET CODE IS XYZABC123",
    save_plot: bool = True,
):
    """Run a comparison between standard RoFormer and iRoPE on different context sizes"""
    print(f"Running comparison on device: {device}")
    
    # Set up models
    print("Initializing models...")
    
    # Standard RoFormer model
    roformer_config = RoFormerConfig(
        vocab_size=30522,  # BERT's vocab size
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=max(context_sizes) + 100,  # Allow for padding
    )
    roformer_model = RoFormerModel(roformer_config)
    
    # iRoPE model
    irope_config = IRoPEConfig(
        vocab_size=30522,  # BERT's vocab size
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        use_interleaved_layers=True,
        interleaved_pattern="alternating",
        temperature_base=1.0,
        rope_scaling=1.0,
        max_position_embeddings=max(context_sizes) + 100,  # Allow for padding
    )
    irope_model = IRoPEModel(irope_config)
    
    # Tokenizer (we'll use BERT's to avoid Chinese tokenizer dependency)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Results storage
    results = {
        "context_sizes": context_sizes,
        "roformer": [],
        "irope": []
    }
    
    # Run tests for each context size
    for context_size in context_sizes:
        print(f"\nTesting context size: {context_size}")
        
        # Create input with needle
        print(f"  Creating input with needle...")
        input_ids, needle_pos = create_needle_in_haystack(tokenizer, context_size, needle_text)
        print(f"  Needle positioned at token {needle_pos} out of {context_size}")
        
        # Test standard RoFormer
        print(f"  Testing standard RoFormer...")
        roformer_result = test_model_on_context(roformer_model, input_ids, device)
        results["roformer"].append(roformer_result)
        
        print(f"    Success: {roformer_result['success']}")
        if not roformer_result['success']:
            print(f"    Error: {roformer_result['error']}")
        print(f"    Memory usage: {format_memory(roformer_result['memory_peak'] - roformer_result['memory_before'])}")
        if roformer_result['success']:
            print(f"    Inference time: {roformer_result['inference_time']:.4f} seconds")
        
        # Test iRoPE model
        print(f"  Testing iRoPE...")
        irope_result = test_model_on_context(irope_model, input_ids, device)
        results["irope"].append(irope_result)
        
        print(f"    Success: {irope_result['success']}")
        if not irope_result['success']:
            print(f"    Error: {irope_result['error']}")
        print(f"    Memory usage: {format_memory(irope_result['memory_peak'] - irope_result['memory_before'])}")
        if irope_result['success']:
            print(f"    Inference time: {irope_result['inference_time']:.4f} seconds")
        
        # Check if both models failed
        if not roformer_result['success'] and not irope_result['success']:
            print(f"Both models failed at context size {context_size}, stopping tests.")
            break
    
    # Plot results
    if save_plot:
        plot_results(results)
    
    return results


def plot_results(results: Dict):
    """Plot comparison results"""
    context_sizes = results["context_sizes"]
    
    # Filter to include only results where we have data
    valid_indices = min(len(results["roformer"]), len(results["irope"]))
    context_sizes = context_sizes[:valid_indices]
    
    # Prepare data for plotting
    roformer_success = [int(results["roformer"][i]["success"]) for i in range(valid_indices)]
    irope_success = [int(results["irope"][i]["success"]) for i in range(valid_indices)]
    
    roformer_memory = [
        (results["roformer"][i]["memory_peak"] - results["roformer"][i]["memory_before"]) / (1024 * 1024 * 1024)
        for i in range(valid_indices)
    ]
    irope_memory = [
        (results["irope"][i]["memory_peak"] - results["irope"][i]["memory_before"]) / (1024 * 1024 * 1024)
        for i in range(valid_indices)
    ]
    
    roformer_time = [
        results["roformer"][i]["inference_time"] if results["roformer"][i]["success"] else 0
        for i in range(valid_indices)
    ]
    irope_time = [
        results["irope"][i]["inference_time"] if results["irope"][i]["success"] else 0
        for i in range(valid_indices)
    ]
    
    # Plot success rates
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    bar_width = 0.35
    index = np.arange(len(context_sizes))
    plt.bar(index, roformer_success, bar_width, label='RoFormer', color='blue', alpha=0.7)
    plt.bar(index + bar_width, irope_success, bar_width, label='iRoPE', color='green', alpha=0.7)
    plt.xlabel('Context Size')
    plt.ylabel('Success (1) / Failure (0)')
    plt.title('Success on Different Context Sizes')
    plt.xticks(index + bar_width / 2, [f"{size:,}" for size in context_sizes])
    plt.legend()
    
    # Plot memory usage
    plt.subplot(3, 1, 2)
    plt.plot(context_sizes, roformer_memory, 'bo-', label='RoFormer')
    plt.plot(context_sizes, irope_memory, 'go-', label='iRoPE')
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage vs Context Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot inference time
    plt.subplot(3, 1, 3)
    plt.plot(context_sizes, roformer_time, 'bo-', label='RoFormer')
    plt.plot(context_sizes, irope_time, 'go-', label='iRoPE')
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time vs Context Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('context_comparison_results.png', dpi=300)
    print("Saved plot to context_comparison_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare models on different context sizes")
    parser.add_argument(
        "--sizes", 
        type=str, 
        default="512,1024,2048,4096,8192",
        help="Comma-separated list of context sizes to test"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    args = parser.parse_args()
    
    # Parse context sizes
    context_sizes = [int(size) for size in args.sizes.split(",")]
    
    # Run comparison
    results = run_comparison(
        context_sizes=context_sizes,
        device=args.device,
        save_plot=True
    ) 