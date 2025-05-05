#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
import glob

def load_test_results():
    """Load all test results from the comparison_results directory"""
    results_dir = "./comparison_results"
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found")
        return []
    
    result_files = glob.glob(f"{results_dir}/comparison_results_*.json")
    all_results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def extract_context_sizes(results):
    """Extract context sizes and success/failure status from results"""
    context_data = []
    
    for result in results:
        if 'target_context_size' in result and 'models' in result:
            size = result['target_context_size']
            success = False
            error = None
            
            if 'llama' in result['models'] and 'error' in result['models']['llama']:
                error = result['models']['llama']['error']
                success = error is None
            
            context_data.append({
                'size': size,
                'success': success,
                'error': error,
                'test_mode': result.get('test_mode', False),
                'timestamp': result.get('timestamp', '')
            })
    
    return context_data

def visualize_context_loading():
    """Create a visualization of context loading tests"""
    results = load_test_results()
    context_data = extract_context_sizes(results)
    
    # Filter to just the test mode results (loading tests)
    loading_tests = [d for d in context_data if d['test_mode']]
    
    # Sort by size
    loading_tests.sort(key=lambda x: x['size'])
    
    # Extract sizes and success status
    sizes = [t['size'] for t in loading_tests]
    success = [1 if t['success'] or t['error'] is None else 0 for t in loading_tests]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot loading success
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(sizes)), success, color=['green' if s else 'red' for s in success])
    plt.xticks(np.arange(len(sizes)), [f"{s:,}" for s in sizes], rotation=45)
    plt.xlabel('Context Size (characters)')
    plt.ylabel('Success (1) / Failure (0)')
    plt.title('Llama 4 Scout Context Loading Tests')
    plt.ylim(0, 1.2)
    
    # Plot loading time if available
    if any('runtime' in result.get('models', {}).get('llama', {}) for result in results):
        runtimes = []
        for size, test in zip(sizes, loading_tests):
            for result in results:
                if (result.get('target_context_size') == size and 
                    result.get('test_mode') and 
                    'llama' in result.get('models', {}) and
                    'runtime' in result.get('models', {}).get('llama', {})):
                    runtimes.append(result['models']['llama']['runtime'])
                    break
            else:
                runtimes.append(0)
        
        plt.subplot(1, 2, 2)
        plt.plot(sizes, runtimes, 'o-', color='blue')
        plt.xlabel('Context Size (characters)')
        plt.ylabel('Loading Time (seconds)')
        plt.title('Llama 4 Scout Loading Time vs Context Size')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('context_loading_results.png', dpi=300)
    plt.close()

def visualize_inference_tests():
    """Create a visualization of inference tests"""
    results = load_test_results()
    context_data = extract_context_sizes(results)
    
    # Filter to just the non-test mode results (inference tests)
    inference_tests = [d for d in context_data if not d['test_mode']]
    
    # Sort by size
    inference_tests.sort(key=lambda x: x['size'])
    
    if not inference_tests:
        print("No inference test results found")
        return
    
    # Extract sizes and error information
    sizes = [t['size'] for t in inference_tests]
    success = [1 if t['success'] or t['error'] is None else 0 for t in inference_tests]
    errors = [t['error'] if not t['success'] and t['error'] else "Success" for t in inference_tests]
    
    # Simplify error messages
    simplified_errors = []
    for error in errors:
        if "CUDA out of memory" in error:
            # Extract the allocation size
            import re
            match = re.search(r"Tried to allocate ([0-9.]+) GiB", error)
            if match:
                simplified_errors.append(f"OOM: {match.group(1)} GiB")
            else:
                simplified_errors.append("CUDA OOM")
        elif error == "Success":
            simplified_errors.append("Success")
        else:
            simplified_errors.append("Error")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot inference success/failure
    plt.bar(np.arange(len(sizes)), success, color=['green' if s else 'red' for s in success])
    plt.xticks(np.arange(len(sizes)), [f"{s:,}" for s in sizes], rotation=45)
    
    # Add error annotations
    for i, (s, e) in enumerate(zip(success, simplified_errors)):
        plt.text(i, 0.5, e, ha='center', va='center', rotation=90, color='white' if not s else 'black')
    
    plt.xlabel('Context Size (characters)')
    plt.ylabel('Success (1) / Failure (0)')
    plt.title('Llama 4 Scout Inference Tests')
    plt.ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('inference_test_results.png', dpi=300)
    plt.close()

def create_memory_usage_graph():
    """Create a visualization of estimated memory usage for different context sizes"""
    # Estimated memory requirements based on our analysis
    context_sizes = [1000, 10000, 100000, 1000000, 10000000]
    model_weights = [80] * len(context_sizes)  # Constant 80GB for model weights with 4-bit quant
    
    # KV cache grows quadratically - these are estimates
    kv_cache = [0.01, 1, 100, 10000, 1000000]  # GB, simplified for visualization
    kv_cache = [min(k, 5000) for k in kv_cache]  # Cap for visualization
    
    # For attention computation during inference (conservative estimates)
    attention_mem = [0.1, 10, 100, 10000, 1000000]  # GB
    attention_mem = [min(a, 5000) for a in attention_mem]  # Cap for visualization
    
    # Create a stacked bar chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(context_sizes))
    width = 0.5
    
    plt.bar(x, model_weights, width, label='Model Weights (4-bit)', color='#1f77b4')
    plt.bar(x, kv_cache, width, bottom=model_weights, label='KV Cache', color='#ff7f0e')
    plt.bar(x, attention_mem, width, bottom=np.array(model_weights) + np.array(kv_cache), 
            label='Attention Computation', color='#2ca02c')
    
    # Add reference lines for different hardware setups
    plt.axhline(y=80, color='r', linestyle='-', alpha=0.5, label='1× H100 (80GB)')
    plt.axhline(y=160, color='g', linestyle='-', alpha=0.5, label='2× H100 (160GB)')
    plt.axhline(y=320, color='b', linestyle='-', alpha=0.5, label='4× H100 (320GB)')
    
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Memory Required (GB)')
    plt.title('Estimated Memory Requirements for Different Context Sizes')
    plt.xticks(x, [f"{s:,}" for s in context_sizes])
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('memory_requirements.png', dpi=300)
    plt.close()

def create_optimization_impact_graph():
    """Create a visualization showing impact of different optimization strategies"""
    context_sizes = [1000, 10000, 100000, 1000000, 10000000]
    
    # Memory requirements with no optimizations (baseline)
    baseline = [80, 90, 180, 10000, 1000000]
    baseline = [min(b, 5000) for b in baseline]
    
    # With Flash Attention 2 (~40% reduction in attention computation)
    flash_attn = [80, 85, 150, 6000, 600000]
    flash_attn = [min(f, 5000) for f in flash_attn]
    
    # With context compression (LLMLingua, ~3× reduction)
    compression = [80, 83, 120, 3500, 350000]
    compression = [min(c, 5000) for c in compression]
    
    # With both Flash Attention 2 and compression
    combined = [80, 82, 100, 2000, 200000]
    combined = [min(c, 5000) for c in combined]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(context_sizes))
    width = 0.2
    
    plt.bar(x - width*1.5, baseline, width, label='No Optimization', color='#d62728')
    plt.bar(x - width/2, flash_attn, width, label='Flash Attention 2', color='#1f77b4')
    plt.bar(x + width/2, compression, width, label='Context Compression', color='#ff7f0e')
    plt.bar(x + width*1.5, combined, width, label='Flash Attn 2 + Compression', color='#2ca02c')
    
    # Add reference lines for different hardware setups
    plt.axhline(y=80, color='r', linestyle='-', alpha=0.3, label='1× H100 (80GB)')
    plt.axhline(y=160, color='r', linestyle='--', alpha=0.3, label='2× H100 (160GB)')
    plt.axhline(y=320, color='r', linestyle=':', alpha=0.3, label='4× H100 (320GB)')
    
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Estimated Memory Required (GB)')
    plt.title('Impact of Optimization Strategies on Memory Requirements')
    plt.xticks(x, [f"{s:,}" for s in context_sizes])
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_impact.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations...")
    visualize_context_loading()
    visualize_inference_tests()
    create_memory_usage_graph()
    create_optimization_impact_graph()
    print("Visualizations saved to:")
    print("- context_loading_results.png")
    print("- inference_test_results.png")
    print("- memory_requirements.png")
    print("- optimization_impact.png") 