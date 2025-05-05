#!/usr/bin/env python3
"""
Llama 4 Scout Context Size Analysis
-----------------------------------
This script analyzes test results across different context sizes to 
visualize memory usage, runtime, and cost efficiency.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Constants
RESULTS_DIR = "./comparison_results"
OUTPUT_DIR = "./analysis_results"
LAMBDA_COST_PER_HOUR = {
    "4xH100": 12.36,  # $12.36/hr for 4x H100 (80GB)
    "8xH100": 23.92,  # $23.92/hr for 8x H100 (80GB)
    "1xH100": 2.49,   # $2.49/hr for 1x H100 (80GB)
}

# Current instance type
CURRENT_INSTANCE = "4xH100"

def load_results(results_dir=RESULTS_DIR):
    """Load all result JSON files from the directory."""
    result_files = glob.glob(os.path.join(results_dir, "*.json"))
    results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add filename to data for reference
                data['filename'] = os.path.basename(file_path)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def extract_metrics(results):
    """Extract key metrics from results into a DataFrame."""
    metrics = []
    
    for result in results:
        # Basic info from each result
        timestamp = result.get('timestamp', 'unknown')
        char_count = result.get('config', {}).get('char_count', 0)
        test_mode = result.get('config', {}).get('test_mode', True)
        
        # Calculate tokens (approx 4 chars per token)
        tokens = char_count // 4
        
        # System info
        gpu_count = 0
        gpu_memory = 0
        total_gpu_memory = 0
        
        if 'system_info' in result:
            sys_info = result['system_info']
            gpu_name = sys_info.get('gpu_name', 'unknown')
            if isinstance(gpu_name, list):
                gpu_count = len(gpu_name)
                if gpu_count > 0 and 'gpu_memory_gb' in sys_info and isinstance(sys_info['gpu_memory_gb'], list):
                    gpu_memory = sys_info['gpu_memory_gb'][0]  # Memory per GPU
                    total_gpu_memory = sum(sys_info['gpu_memory_gb'])
            else:
                gpu_count = 1 if sys_info.get('cuda_available', False) else 0
                gpu_memory = sys_info.get('gpu_memory_gb', 0)
                total_gpu_memory = gpu_memory
            
        # Process each model result
        for model_result in result.get('results', []):
            model_name = model_result.get('model', 'unknown')
            runtime = model_result.get('runtime_seconds', 0)
            success = model_result.get('success', False)
            error = model_result.get('error', None)
            is_test_mode = model_result.get('test_mode', test_mode)
            
            # Calculate cost
            hourly_cost = LAMBDA_COST_PER_HOUR.get(CURRENT_INSTANCE, 0)
            runtime_hours = runtime / 3600  # Convert seconds to hours
            inference_cost = runtime_hours * hourly_cost
            
            # Create metric entry
            metrics.append({
                'timestamp': timestamp,
                'filename': result.get('filename', ''),
                'model': model_name,
                'char_count': char_count,
                'tokens': tokens,
                'runtime_seconds': runtime,
                'success': success,
                'test_mode': is_test_mode,
                'error': error is not None,
                'error_message': error,
                'gpu_count': gpu_count,
                'gpu_memory_per_device_gb': gpu_memory,
                'total_gpu_memory_gb': total_gpu_memory,
                'hourly_cost': hourly_cost,
                'inference_cost': inference_cost,
                'tokens_per_second': tokens / max(runtime, 0.001),
                'cost_per_million_tokens': (inference_cost * 1_000_000) / max(tokens, 1)
            })
    
    # Convert to DataFrame and sort by context size
    df = pd.DataFrame(metrics)
    if not df.empty:
        df = df.sort_values('tokens')
    
    return df

def plot_runtime_vs_context(df, output_dir=OUTPUT_DIR):
    """Plot runtime vs context size."""
    if df.empty:
        print("No data to plot runtime vs context size.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Filter for only test runs without errors
    plot_df = df[(~df['error']) & (df['model'] == 'Llama 4 Scout')]
    
    # Use different markers for test mode vs full inference
    test_mask = plot_df['test_mode']
    
    # Plot test mode runs
    if any(test_mask):
        sns.lineplot(
            data=plot_df[test_mask], 
            x='tokens', 
            y='runtime_seconds',
            marker='o',
            label='Test Mode (No Inference)'
        )
    
    # Plot full inference runs
    if any(~test_mask):
        sns.lineplot(
            data=plot_df[~test_mask], 
            x='tokens', 
            y='runtime_seconds',
            marker='x',
            label='Full Inference'
        )
    
    plt.title('Runtime vs Context Size (Llama 4 Scout)')
    plt.xlabel('Tokens')
    plt.ylabel('Runtime (seconds)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'runtime_vs_context.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Runtime vs context size plot saved to {output_path}")

def plot_cost_vs_context(df, output_dir=OUTPUT_DIR):
    """Plot estimated cost vs context size."""
    if df.empty:
        print("No data to plot cost vs context size.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Filter for only test runs without errors
    plot_df = df[(~df['error']) & (df['model'] == 'Llama 4 Scout')]
    
    # Use different markers for test mode vs full inference
    test_mask = plot_df['test_mode']
    
    # Plot test mode runs
    if any(test_mask):
        sns.lineplot(
            data=plot_df[test_mask], 
            x='tokens', 
            y='cost_per_million_tokens',
            marker='o',
            label='Test Mode (No Inference)'
        )
    
    # Plot full inference runs
    if any(~test_mask):
        sns.lineplot(
            data=plot_df[~test_mask],

            x='tokens', 
            y='cost_per_million_tokens',
            marker='x',
            label='Full Inference'
        )
    
    plt.title(f'Cost per Million Tokens vs Context Size (Llama 4 Scout)\nInstance: {CURRENT_INSTANCE} at ${LAMBDA_COST_PER_HOUR.get(CURRENT_INSTANCE, 0)}/hr')
    plt.xlabel('Context Size (Tokens)')
    plt.ylabel('Cost per Million Tokens ($)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'cost_vs_context.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Cost vs context size plot saved to {output_path}")

def plot_tokens_per_second(df, output_dir=OUTPUT_DIR):
    """Plot tokens per second vs context size."""
    if df.empty:
        print("No data to plot tokens per second vs context size.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Filter for only test runs without errors
    plot_df = df[(~df['error']) & (df['model'] == 'Llama 4 Scout')]
    
    # Use different markers for test mode vs full inference
    test_mask = plot_df['test_mode']
    
    # Plot test mode runs
    if any(test_mask):
        sns.lineplot(
            data=plot_df[test_mask], 
            x='tokens', 
            y='tokens_per_second',
            marker='o',
            label='Test Mode (No Inference)'
        )
    
    # Plot full inference runs
    if any(~test_mask):
        sns.lineplot(
            data=plot_df[~test_mask], 
            x='tokens', 
            y='tokens_per_second',
            marker='x',
            label='Full Inference'
        )
    
    plt.title('Processing Speed vs Context Size (Llama 4 Scout)')
    plt.xlabel('Context Size (Tokens)')
    plt.ylabel('Tokens per Second')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'tokens_per_second.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Tokens per second plot saved to {output_path}")

def plot_success_rate(df, output_dir=OUTPUT_DIR):
    """Plot success rate vs context size for full inference runs."""
    if df.empty:
        print("No data to plot success rate vs context size.")
        return
    
    # Filter for only full inference runs without errors
    plot_df = df[(~df['error']) & (~df['test_mode']) & (df['model'] == 'Llama 4 Scout')]
    
    if plot_df.empty:
        print("No full inference runs to plot success rate.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Group by token count and calculate success rate
    success_rate = plot_df.groupby('tokens')['success'].mean().reset_index()
    
    # Plot success rate
    sns.lineplot(
        data=success_rate,
        x='tokens',
        y='success',
        marker='o'
    )
    
    plt.title('Success Rate vs Context Size (Llama 4 Scout)')
    plt.xlabel('Context Size (Tokens)')
    plt.ylabel('Success Rate')
    plt.xscale('log')
    plt.ylim(0, 1.1)  # Success rate is between 0 and 1
    plt.grid(True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'success_rate.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Success rate plot saved to {output_path}")

def generate_summary_report(df, output_dir=OUTPUT_DIR):
    """Generate a summary report of all tests."""
    if df.empty:
        print("No data to generate summary report.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for Llama 4 Scout model
    llama_df = df[df['model'] == 'Llama 4 Scout'].copy()
    
    # Extract test results organized by context size
    test_results = llama_df.sort_values('tokens')
    
    # Generate summary table as DataFrame
    summary = test_results[['tokens', 'runtime_seconds', 'success', 'test_mode', 'error', 'cost_per_million_tokens']]
    summary['context_size'] = summary['tokens'].apply(lambda x: f"{x:,}")
    summary['runtime'] = summary['runtime_seconds'].apply(lambda x: f"{x:.2f}s")
    summary['cost_per_1M_tokens'] = summary['cost_per_million_tokens'].apply(lambda x: f"${x:.2f}")
    summary['mode'] = summary['test_mode'].apply(lambda x: "Test Only" if x else "Full Inference")
    summary['result'] = summary.apply(
        lambda row: "Error" if row['error'] else ("Success" if row['success'] else "Failed to find needle"), 
        axis=1
    )
    
    # Select and order columns for display
    display_summary = summary[['context_size', 'mode', 'runtime', 'cost_per_1M_tokens', 'result']]
    
    # Write summary to file
    output_path = os.path.join(output_dir, 'summary_report.md')
    with open(output_path, 'w') as f:
        f.write("# Llama 4 Scout Context Size Analysis\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Instance: {CURRENT_INSTANCE} (${LAMBDA_COST_PER_HOUR.get(CURRENT_INSTANCE, 0)}/hr)\n\n")
        f.write("## Summary of Tests\n\n")
        f.write(display_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Analysis\n\n")
        
        # Calculate some statistics
        if not llama_df.empty:
            max_tokens = llama_df['tokens'].max()
            max_runtime = llama_df['runtime_seconds'].max()
            mean_cost = llama_df['cost_per_million_tokens'].mean()
            success_runs = llama_df[(~llama_df['test_mode']) & (~llama_df['error'])]
            success_rate = success_runs['success'].mean() if not success_runs.empty else "N/A"
            
            f.write(f"- Maximum context size tested: {max_tokens:,} tokens\n")
            f.write(f"- Maximum runtime: {max_runtime:.2f} seconds\n")
            f.write(f"- Average cost per million tokens: ${mean_cost:.2f}\n")
            f.write(f"- Success rate on inference runs: {success_rate}\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("- For production use, contexts of up to [X] tokens are most cost-effective\n")
        f.write("- Cost scales approximately [linearly/sub-linearly/super-linearly] with context size\n")
        f.write("- The 4x H100 setup is [sufficient/insufficient] for handling the full 10M token context window\n")
    
    print(f"Summary report saved to {output_path}")

def main():
    # Load result files
    print("Loading test results...")
    results = load_results()
    print(f"Loaded {len(results)} result files.")
    
    if not results:
        print("No results found to analyze.")
        return
    
    # Extract metrics
    print("Extracting metrics...")
    metrics_df = extract_metrics(results)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_runtime_vs_context(metrics_df)
    plot_cost_vs_context(metrics_df)
    plot_tokens_per_second(metrics_df)
    plot_success_rate(metrics_df)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(metrics_df)
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 