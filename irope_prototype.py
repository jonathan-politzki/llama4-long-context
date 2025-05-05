#!/usr/bin/env python3
"""
iRoPE Architecture Prototype

This is a conceptual implementation of Meta's iRoPE (interleaved Rotary Position Embedding)
architecture described in their Llama 4 announcement. This is NOT an actual working
implementation but rather a demonstration of the key concepts.

Key features:
1. Interleaved attention layers (some with RoPE, some without positional encoding)
2. Temperature scaling for attention
3. Modified RoPE implementation for better length extrapolation

References:
- Meta's Llama 4 announcement (April 2025)
- RoFormer paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Implementation of Rotary Positional Embedding (RoPE)
    with modifications for better length generalization.
    """
    def __init__(
        self, 
        dim: int, 
        base: int = 10000, 
        scale_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_factor = scale_factor  # Scaling factor for better length extrapolation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        batch_size, num_heads, seq_length, hidden_size = x.shape
        
        if seq_len is None:
            seq_len = seq_length
        
        # Apply scaling for better extrapolation to longer sequences
        # Create position indices tensor [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).float()
        positions = positions / self.scale_factor  # Apply scaling
        
        # Create sinusoidal pattern for each position and frequency
        # Shape: [seq_len, dim//2]
        freqs = torch.outer(positions, self.inv_freq)
        
        # Create complex-valued embeddings e^(i * freqs)
        # [seq_len, hidden_size//2] -> [seq_len, hidden_size]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Create sin and cos patterns
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, hidden_size]
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, hidden_size]
        
        # Apply rotations using complex number multiplication formulas
        # This handles broadcasting correctly
        return x * cos + self._rotate_half(x) * sin


class AttentionWithRoPE(nn.Module):
    """
    Attention layer with Rotary Positional Embedding
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        rope_scaling: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=self.head_dim,
            scale_factor=rope_scaling  # Scale factor for length extrapolation
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project inputs
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to multiple heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply temperature scaling
        attention_scores = attention_scores / temperature
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        
        # Final projection
        output = self.o_proj(context)
        
        return output


class AttentionWithoutPositionalEncoding(nn.Module):
    """
    Attention layer without any positional encoding - 
    for global, position-agnostic attention
    """
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project inputs
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to multiple heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores without positional bias
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply temperature scaling
        attention_scores = attention_scores / temperature
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        
        # Final projection
        output = self.o_proj(context)
        
        return output


class IRoPELayer(nn.Module):
    """
    Transformer layer with iRoPE architecture
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        layer_idx: int,
        use_rope: bool = True, 
        rope_scaling: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        
        # Choose attention mechanism based on layer type
        if use_rope:
            self.attention = AttentionWithRoPE(
                hidden_size=hidden_size, 
                num_heads=num_heads,
                rope_scaling=rope_scaling
            )
        else:
            self.attention = AttentionWithoutPositionalEncoding(
                hidden_size=hidden_size, 
                num_heads=num_heads
            )
            
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        # Pre-normalization for attention
        normed_hidden_states = self.attn_norm(hidden_states)
        
        # Apply attention
        attn_output = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            temperature=temperature
        )
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Pre-normalization for MLP
        normed_hidden_states = self.mlp_norm(hidden_states)
        
        # Apply MLP
        mlp_output = self.mlp(normed_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states


class IRoPEModel(nn.Module):
    """
    Complete model with interleaved RoPE and no-position layers
    """
    def __init__(
        self, 
        vocab_size: int = 32000,
        hidden_size: int = 768, 
        num_heads: int = 12,
        num_layers: int = 12,
        rope_layers_pattern: str = "alternating",  # "alternating" or "custom"
        rope_scaling: float = 1.0,
        temperature_base: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temperature_base = temperature_base
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Determine which layers use RoPE
        self.rope_layers = []
        if rope_layers_pattern == "alternating":
            # Every other layer uses RoPE
            self.rope_layers = [i % 2 == 0 for i in range(num_layers)]
        elif rope_layers_pattern == "custom":
            # Use RoPE in early layers but not later layers
            # This is just an example pattern
            self.rope_layers = [i < (num_layers // 2) for i in range(num_layers)]
        else:
            # Default to all layers using RoPE
            self.rope_layers = [True] * num_layers
        
        # Create layers
        self.layers = nn.ModuleList([
            IRoPELayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                layer_idx=i,
                use_rope=self.rope_layers[i],
                rope_scaling=rope_scaling
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def get_temperature_for_length(self, seq_length: int) -> float:
        """
        Compute temperature scaling factor based on sequence length
        
        This is a simplified version of what might be happening in 
        Meta's temperature scaling - the exact formula is unknown.
        """
        # A simple formula that increases temperature with sequence length
        # This helps control attention for very long sequences
        temperature = self.temperature_base * (1.0 + math.log(max(1, seq_length / 256)) / 10)
        return temperature
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # Compute embeddings
        hidden_states = self.token_embeddings(input_ids)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Prepare causal attention mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask * -1e9  # Set masked positions to large negative value
        
        # Get temperature based on sequence length
        temperature = self.get_temperature_for_length(seq_length)
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                temperature=temperature
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return logits


def demo_irope_model():
    """
    Demonstrate the iRoPE model with a simple example
    """
    # Create a small model
    model = IRoPEModel(
        vocab_size=32000,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        rope_layers_pattern="alternating",
        rope_scaling=1.0,
        temperature_base=1.0
    )
    
    # Print model structure
    print(f"iRoPE Model Structure:")
    print(f"- Vocab size: {model.vocab_size}")
    print(f"- Hidden size: {model.hidden_size}")
    print(f"- Number of heads: {model.num_heads}")
    print(f"- Number of layers: {model.num_layers}")
    print(f"- RoPE layers pattern: {model.rope_layers}")
    
    # Print temperature scaling for different sequence lengths
    print("\nTemperature Scaling:")
    for length in [256, 1024, 4096, 16384, 65536, 262144, 1048576, 10485760]:
        temp = model.get_temperature_for_length(length)
        print(f"- Length {length:,}: temperature = {temp:.4f}")
    
    # Create a small input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"\nOutput shape: {output.shape}")
    print("Successfully processed through iRoPE model!")
    
    # Try a longer sample to test extrapolation
    print("\nTesting with a longer sequence...")
    long_seq_length = 1024
    long_input_ids = torch.randint(0, model.vocab_size, (1, long_seq_length))
    with torch.no_grad():
        long_output = model(long_input_ids)
    print(f"Longer output shape: {long_output.shape}")
    print("Successfully processed longer sequence!")


if __name__ == "__main__":
    demo_irope_model() 