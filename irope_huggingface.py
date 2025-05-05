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
    RoFormerEmbeddings,
    RoFormerSinusoidalPositionalEmbedding
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
        sinusoidal_pos=None,
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
        if self.use_rope and sinusoidal_pos is not None:
            # Apply RoFormer's rotary embeddings
            sinusoidal_pos = sinusoidal_pos[:, :, : query_layer.shape[-2] * 2, :]
            sin, cos = torch.chunk(sinusoidal_pos, 2, dim=-1)
            sin_pos = torch.stack([sin, sin], dim=-1).reshape(sinusoidal_pos.shape)  # (batch_size, num_heads, seq_len, head_dim)
            cos_pos = torch.stack([cos, cos], dim=-1).reshape(sinusoidal_pos.shape)  # (batch_size, num_heads, seq_len, head_dim)
            
            # Rotary embeddings
            query_layer = self._apply_rotary(query_layer, sin_pos, cos_pos)
            key_layer = self._apply_rotary(key_layer, sin_pos, cos_pos)
            
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
        
    def _apply_rotary(self, x, sin, cos):
        # Apply rotary embeddings
        sin = sin[:, :, :x.shape[2], :]
        cos = cos[:, :, :x.shape[2], :]
        
        # Split the input tensor on the hidden dimension
        x_shape = x.shape
        half_dim = x.shape[-1] // 2
        
        # Get the first half and second half of the hidden dimension
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        # Repeat the sinusoidal embeddings to match the hidden dimension if needed
        if sin.shape[-1] != half_dim:
            repeat_factor = half_dim // sin.shape[-1]
            sin = sin.repeat(1, 1, 1, repeat_factor)
            cos = cos.repeat(1, 1, 1, repeat_factor)
            
        # Apply the rotary embeddings
        rotated_x = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated_x


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
            
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Self-attention with positional embedding if appropriate for this layer
        layer_outputs = self.attention(
            self.LayerNorm(hidden_states),
            attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = layer_outputs[0]
        outputs = layer_outputs[1:]  # add self attentions if we output attention weights

        # Feed forward
        hidden_states = hidden_states + self.dropout(attention_output)
        intermediate_output = self.intermediate(self.LayerNorm(hidden_states))
        if isinstance(self.intermediate_act_fn, str):
            intermediate_output = getattr(F, self.intermediate_act_fn)(intermediate_output)
        else:
            intermediate_output = self.intermediate_act_fn(intermediate_output)
        hidden_states = hidden_states + self.dropout(self.output(intermediate_output))
        
        outputs = (hidden_states,) + outputs
        return outputs


class IRoPEEncoder(nn.Module):
    """
    Modified RoFormerEncoder that uses our custom IRoPELayer and handles positional embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [IRoPELayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        
        # Initialize positional embeddings
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads // 2,
            padding_idx=config.pad_token_id,
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None

        # Generate positional embeddings
        seq_length = hidden_states.shape[1]
        past_key_values_length = 0
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], past_key_values_length)[None, None, :, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                sinusoidal_pos=sinusoidal_pos,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )


class IRoPEModel(RoFormerModel):
    """
    Modified RoFormerModel that uses our custom IRoPEEncoder.
    """
    config_class = IRoPEConfig

    def __init__(self, config):
        # Skip the parent constructor and use the grandparent to avoid unwanted initializations
        nn.Module.__init__(self)
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
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        
        return_dict = self.config.use_return_dict
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
            
        # Create a proper return object
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs[1] if output_hidden_states else None,
            attentions=encoder_outputs[2] if output_attentions else None,
            cross_attentions=encoder_outputs[3] if output_attentions and encoder_hidden_states is not None else None,
        )


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
        max_position_embeddings=4096,  # Support longer positions
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
    if hasattr(model.encoder.layer[0].attention, "get_temperature_for_length"):
        print("\nTemperature Scaling:")
        for length in [256, 1024, 4096, 16384, 65536, 262144, 1048576, 10485760]:
            temp = model.encoder.layer[0].attention.get_temperature_for_length(length)
            print(f"- Length {length:,}: temperature = {temp:.4f}")
    
    # Test with a small input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    print(f"\nTesting with sequence length: {seq_length}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print("Successfully processed through iRoPE model!")
    
    # Test with a longer sequence
    long_seq_length = 1024
    print(f"\nTesting with longer sequence: {long_seq_length}")
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