"""
Training Visualization Script

Visualizes training progress including loss curves and learning rate schedules.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import json
import matplotlib.pyplot as plt
import argparse
import os


def visualize_training(log_file, save_dir='./visualizations'):
    """
    Visualize training progress from log file
    
    Author: Molla Samser (https://rskworld.in)
    """
    # Load training log
    with open(log_file, 'r') as f:
        log = json.load(f)
    
    epochs = log.get('epochs', [])
    train_loss = log.get('train_loss', [])
    val_loss = log.get('val_loss', [])
    learning_rates = log.get('learning_rates', [])
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    if learning_rates:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, learning_rates, label='Learning Rate', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Author: Molla Samser - https://rskworld.in")
    print(f"Training visualization saved to: {save_path}")
    
    # Print summary
    if train_loss and val_loss:
        print(f"\nTraining Summary:")
        print(f"  Final train loss: {train_loss[-1]:.4f}")
        print(f"  Final validation loss: {val_loss[-1]:.4f}")
        print(f"  Best validation loss: {min(val_loss):.4f} (epoch {epochs[val_loss.index(min(val_loss))]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training log JSON file')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_training(args.log_file, args.save_dir)

