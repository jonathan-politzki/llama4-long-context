#!/usr/bin/env python3
"""
iRoPE Architecture Implementation using Hugging Face's RoFormer

This implementation adapts Meta's iRoPE (interleaved Rotary Position Embedding)
architecture described in their Llama 4 announcement, using Hugging Face's RoFormer
as the foundation.

Key features:
1. Interleaved attention layers (some with RoPE, some without positional encoding)
2. Temperature scaling for attention
3. Using RoFormer's optimized RoPE implementation

References:
- Meta's Llama 4 announcement (April 2025)
- RoFormer paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Hugging Face RoFormer implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# Import Hugging Face's RoFormer
from transformers import RoFormerModel, RoFormerConfig
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfAttention,
    RoFormerLayer,
    RoFormerEncoder,
    RoFormerEmbeddings
)


class IRoPEConfig(RoFormerConfig):
    """
    Configuration class for iRoPE model.
    Extends RoFormerConfig with additional parameters for interleaved layers and temperature scaling.
    """
    def __init__(
        self,
        use_interleaved_layers=True,
        interleaved_pattern="alternating",  # "alternating", "custom"
        custom_rope_layers=None,
        temperature_base=1.0,
        rope_scaling=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_interleaved_layers = use_interleaved_layers
        self.interleaved_pattern = interleaved_pattern
        self.custom_rope_layers = custom_rope_layers
        self.temperature_base = temperature_base
        self.rope_scaling = rope_scaling
        self.model_type = "irope"


class IRoPESelfAttention(RoFormerSelfAttention):
    """
    Modified self-attention layer that can disable positional encoding and apply temperature scaling.
    """
    def __init__(self, config, layer_idx=0):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.temperature_base = getattr(config, "temperature_base", 1.0)
        
        # Determine if this layer should use RoPE based on the interleaved pattern
        self.use_rope = True  # Default to using RoPE
        
        if getattr(config, "use_interleaved_layers", False):
            if config.interleaved_pattern == "alternating":
                self.use_rope = layer_idx % 2 == 0
            elif config.interleaved_pattern == "custom" and config.custom_rope_layers is not None:
                if layer_idx < len(config.custom_rope_layers):
                    self.use_rope = config.custom_rope_layers[layer_idx]

    def get_temperature_for_length(self, seq_length):
        """
        Compute temperature scaling factor based on sequence length.
        
        This is a simplified version of what might be happening in 
        Meta's temperature scaling - the exact formula is unknown.
        """
        # A simple formula that increases temperature with sequence length
        # This helps control attention for very long sequences
        temperature = self.temperature_base * (1.0 + math.log(max(1, seq_length / 256)) / 10)
        return temperature

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is cross-attention, we need to process encoder states
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Apply RoPE only if this layer is configured to use it
        if self.use_rope:
            # Get seq_len for temperature scaling
            seq_length = hidden_states.size(1)
            
            # Apply RoFormer's rotary embeddings
            query_layer, key_layer = self.apply_rotary_position_embeddings(
                self.rotary_embeddings, query_layer, key_layer
            )
            
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply temperature scaling
        seq_length = hidden_states.size(1)
        temperature = self.get_temperature_for_length(seq_length)
        attention_scores = attention_scores / temperature
        
        # Apply attention mask
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs


class IRoPELayer(RoFormerLayer):
    """
    Modified RoFormerLayer that uses our custom IRoPESelfAttention.
    """
    def __init__(self, config, layer_idx=0):
        # Skip the parent constructor and use the grandparent
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = IRoPESelfAttention(config, layer_idx=layer_idx)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = getattr(F, config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act


class IRoPEEncoder(RoFormerEncoder):
    """
    Modified RoFormerEncoder that uses our custom IRoPELayer.
    """
    def __init__(self, config):
        super(RoFormerEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [IRoPELayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False


class IRoPEModel(RoFormerModel):
    """
    Modified RoFormerModel that uses our custom IRoPEEncoder.
    """
    config_class = IRoPEConfig

    def __init__(self, config):
        super(RoFormerModel, self).__init__(config)
        self.config = config
        self.embeddings = RoFormerEmbeddings(config)
        self.encoder = IRoPEEncoder(config)

        # Initialize weights
        self.post_init()
    
    def get_rope_pattern(self):
        """
        Return the pattern of which layers are using RoPE.
        """
        pattern = []
        for layer_module in self.encoder.layer:
            pattern.append(layer_module.attention.use_rope)
        return pattern


def demo_irope_model():
    """
    Demonstrate how to use the iRoPE model.
    """
    # Create config
    config = IRoPEConfig(
        vocab_size=30522,  # BERT's vocab size
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        use_interleaved_layers=True,
        interleaved_pattern="alternating",
        temperature_base=1.0,
        rope_scaling=1.0,
    )
    
    # Create model
    model = IRoPEModel(config)
    
    # Show model structure
    print(f"iRoPE Model Structure:")
    print(f"- Vocab size: {model.config.vocab_size}")
    print(f"- Hidden size: {model.config.hidden_size}")
    print(f"- Number of heads: {model.config.num_attention_heads}")
    print(f"- Number of layers: {model.config.num_hidden_layers}")
    print(f"- RoPE layers pattern: {model.get_rope_pattern()}")
    
    # Show temperature scaling for different sequence lengths
    for length in [256, 1024, 4096, 16384, 65536, 262144, 1048576, 10485760]:
        temp = model.encoder.layer[0].attention.get_temperature_for_length(length)
        print(f"- Length {length:,}: temperature = {temp:.4f}")
    
    # Test with a small input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    print(f"\nOutput shape: {outputs.last_hidden_state.shape}")
    print("Successfully processed through iRoPE model!")
    
    # Test with a longer sequence
    print("\nTesting with a longer sequence...")
    long_seq_length = 1024
    long_input_ids = torch.randint(0, model.config.vocab_size, (1, long_seq_length))
    long_attention_mask = torch.ones(1, long_seq_length)
    
    with torch.no_grad():
        long_outputs = model(long_input_ids, attention_mask=long_attention_mask)
    
    print(f"Longer output shape: {long_outputs.last_hidden_state.shape}")
    print("Successfully processed longer sequence!")


if __name__ == "__main__":
    print("Loading required libraries...")
    # Check if transformers is installed
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not found. Please install it with: pip install transformers")
        exit(1)
    
    print("\nRunning iRoPE demo...")
    demo_irope_model() 