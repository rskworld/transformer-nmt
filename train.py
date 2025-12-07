"""
Training Script for Transformer-based Neural Machine Translation

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training loop with optimization
- Model checkpointing
- Loss tracking and logging

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from tqdm import tqdm

from transformer_model import Transformer
from data_preprocessing import (
    load_data, create_data_loader, Vocabulary,
    save_vocabularies, normalize_string
)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization
    
    Author: Molla Samser (https://rskworld.in)
    """
    
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None
    
    def forward(self, pred, target):
        assert pred.size(1) == self.vocab_size
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


def train_epoch(model, dataloader, criterion, optimizer, device, clip=1.0):
    """
    Train for one epoch
    
    Author: Molla Samser (https://rskworld.in)
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Author: Molla Samser (https://rskworld.in)
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(
    data_path,
    num_epochs=50,
    batch_size=32,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    dropout=0.1,
    lr=0.0001,
    max_length=100,
    min_freq=2,
    num_pairs=None,
    save_dir='./models',
    device=None
):
    """
    Main training function
    
    Author: Molla Samser (https://rskworld.in)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Author: Molla Samser - https://rskworld.in")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    src_sentences, tgt_sentences = load_data(data_path, num_pairs)
    
    print(f"Loaded {len(src_sentences)} sentence pairs")
    
    # Split into train and validation
    split_idx = int(0.9 * len(src_sentences))
    train_src = src_sentences[:split_idx]
    train_tgt = tgt_sentences[:split_idx]
    val_src = src_sentences[split_idx:]
    val_tgt = tgt_sentences[split_idx:]
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary('source')
    tgt_vocab = Vocabulary('target')
    
    src_vocab.build_vocabulary(train_src, min_freq)
    tgt_vocab.build_vocabulary(train_tgt, min_freq)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Save vocabularies
    save_vocabularies(src_vocab, tgt_vocab, save_dir)
    
    # Create data loaders
    train_loader = create_data_loader(
        train_src, train_tgt, src_vocab, tgt_vocab,
        batch_size, max_length, shuffle=True
    )
    val_loader = create_data_loader(
        val_src, val_tgt, src_vocab, tgt_vocab,
        batch_size, max_length, shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=max_length,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epochs': []
    }
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log training progress
        training_log['train_loss'].append(train_loss)
        training_log['val_loss'].append(val_loss)
        training_log['learning_rates'].append(current_lr)
        training_log['epochs'].append(epoch + 1)
        
        # Save training log
        log_path = os.path.join(save_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'd_model': d_model,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'd_ff': d_ff,
                'dropout': dropout,
                'max_length': max_length,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in: {save_dir}")
    print(f"Training log saved to: {log_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transformer NMT Model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='Minimum word frequency')
    parser.add_argument('--num_pairs', type=int, default=None,
                       help='Number of sentence pairs to use')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        max_length=args.max_length,
        min_freq=args.min_freq,
        num_pairs=args.num_pairs,
        save_dir=args.save_dir
    )

