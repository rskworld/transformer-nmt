"""
Utility Functions for Transformer NMT

Helper functions for visualization, evaluation, and other utilities.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer_model import Transformer


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Author: Molla Samser (https://rskworld.in)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_attention(attention_weights, src_tokens, tgt_tokens, head_idx=0):
    """
    Visualize attention weights
    
    Author: Molla Samser (https://rskworld.in)
    
    Args:
        attention_weights: Attention weights tensor
        src_tokens: Source token list
        tgt_tokens: Target token list
        head_idx: Which attention head to visualize
    """
    if attention_weights.dim() == 4:
        # Multi-head attention: [batch, heads, seq_len, seq_len]
        attn = attention_weights[0, head_idx].cpu().detach().numpy()
    else:
        attn = attention_weights[0].cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title(f'Attention Weights (Head {head_idx})')
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    plt.tight_layout()
    return plt


def get_model_size_mb(model):
    """
    Calculate model size in megabytes
    
    Author: Molla Samser (https://rskworld.in)
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def create_sample_model(src_vocab_size=5000, tgt_vocab_size=5000, **kwargs):
    """
    Create a sample transformer model with default or custom parameters
    
    Author: Molla Samser (https://rskworld.in)
    """
    defaults = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'max_len': 5000,
        'dropout': 0.1
    }
    defaults.update(kwargs)
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **defaults
    )
    return model


def print_model_summary(model):
    """
    Print a summary of the model architecture
    
    Author: Molla Samser (https://rskworld.in)
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print("\nModel Architecture:")
    print(model)
    print("=" * 60)


def save_model_info(model, save_path):
    """
    Save model information to a text file
    
    Author: Molla Samser (https://rskworld.in)
    """
    with open(save_path, 'w') as f:
        f.write("Transformer NMT Model Information\n")
        f.write("=" * 60 + "\n")
        f.write(f"Author: Molla Samser (https://rskworld.in)\n")
        f.write(f"Total parameters: {count_parameters(model):,}\n")
        f.write(f"Model size: {get_model_size_mb(model):.2f} MB\n")
        f.write("\nModel Architecture:\n")
        f.write(str(model))
        f.write("\n" + "=" * 60 + "\n")


def format_time(seconds):
    """
    Format seconds into hours, minutes, seconds
    
    Author: Molla Samser (https://rskworld.in)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

