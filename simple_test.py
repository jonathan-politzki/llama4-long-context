#!/usr/bin/env python3
"""
Simple iRoPE Test Script

This is a minimalist script to demonstrate the key ideas of iRoPE:
1. Interleaved attention layers (some with RoPE, some without positional encoding)
2. Temperature scaling based on sequence length

This uses only standard HuggingFace components to avoid compatibility issues.
"""

import torch
from transformers import BertTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
import numpy as np

# Define a simple function to simulate temperature scaling
def get_temperature(seq_length, base=1.0):
    """Temperature increases logarithmically with sequence length"""
    return base * (1.0 + np.log(max(1, seq_length / 256)) / 10)

# Create a sequence of lengths to test
lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]

# Calculate temperatures for each length
temperatures = [get_temperature(length) for length in lengths]

# Plot the temperature scaling
plt.figure(figsize=(10, 6))
plt.plot(lengths, temperatures, 'bo-')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.xlabel('Sequence Length (tokens)')
plt.ylabel('Temperature Factor')
plt.title('iRoPE Temperature Scaling')

for i, (length, temp) in enumerate(zip(lengths, temperatures)):
    if i % 2 == 0:  # Label every other point
        plt.annotate(f"{length}: {temp:.2f}", (length, temp), 
                    textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('irope_temperature_scaling.png')
print("Generated temperature scaling plot at irope_temperature_scaling.png")

# Now demonstrate the concept of interleaved layers
print("\nDemonstrating interleaved attention layers:")
num_layers = 12
interleaved_pattern = [i % 2 == 0 for i in range(num_layers)]  # Alternating True/False

print(f"Layer pattern (True = use RoPE, False = no positional encoding):")
for i, uses_rope in enumerate(interleaved_pattern):
    print(f"  Layer {i}: {'✓' if uses_rope else '✗'} positional encoding")

# Test a simple forward pass with standard BERT (just to verify tokenization works)
print("\nTesting tokenization and model loading with standard BERT:")
try:
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Create a simple input
    text = "This is a test sentence to verify the tokenization works correctly."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Successfully ran inference with sequence length {inputs['input_ids'].shape[1]}")
    print(f"  Input text: '{text}'")
    print(f"  Output shape: {outputs.last_hidden_state.shape}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")

print("\nSimple test complete!") 