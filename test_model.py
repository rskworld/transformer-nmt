"""
Test Script for Transformer NMT Model

This script tests the transformer model with dummy data to ensure
all components are working correctly.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
from transformer_model import Transformer
from data_preprocessing import Vocabulary
from utils import print_model_summary, count_parameters


def test_model_forward():
    """
    Test forward pass of the transformer model
    
    Author: Molla Samser (https://rskworld.in)
    """
    print("Testing Transformer Model")
    print("=" * 60)
    print("Author: Molla Samser (https://rskworld.in)\n")
    
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    batch_size = 4
    src_len = 10
    tgt_len = 12
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=100,
        dropout=0.1
    )
    
    print_model_summary(model)
    
    # Create dummy data
    print("\nCreating dummy data...")
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt[:, :-1])  # Exclude last token for input
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {tgt_len-1}, {tgt_vocab_size}]")
    
    # Check output shape
    assert output.shape == (batch_size, tgt_len - 1, tgt_vocab_size), \
        f"Output shape mismatch! Got {output.shape}, expected {(batch_size, tgt_len - 1, tgt_vocab_size)}"
    
    print("\n✓ Forward pass successful!")
    
    # Test loss calculation
    print("\nTesting loss calculation...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    output_flat = output.reshape(-1, tgt_vocab_size)
    target_flat = tgt[:, 1:].reshape(-1)  # Shift by 1 for next token prediction
    
    loss = criterion(output_flat, target_flat)
    print(f"Loss: {loss.item():.4f}")
    print("✓ Loss calculation successful!")
    
    # Test with different batch sizes
    print("\nTesting with different batch sizes...")
    for bs in [1, 2, 8]:
        src_test = torch.randint(1, src_vocab_size, (bs, src_len))
        tgt_test = torch.randint(1, tgt_vocab_size, (bs, tgt_len))
        
        with torch.no_grad():
            output_test = model(src_test, tgt_test[:, :-1])
        
        assert output_test.shape[0] == bs, f"Batch size mismatch for batch_size={bs}"
        print(f"✓ Batch size {bs} works correctly")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_vocabulary():
    """
    Test vocabulary functionality
    
    Author: Molla Samser (https://rskworld.in)
    """
    print("\nTesting Vocabulary...")
    print("-" * 60)
    
    vocab = Vocabulary('test')
    sentences = [
        "hello world",
        "hello python",
        "world peace",
        "python programming"
    ]
    
    vocab.build_vocabulary(sentences, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: PAD={vocab.PAD_token}, SOS={vocab.SOS_token}, EOS={vocab.EOS_token}, UNK={vocab.UNK_token}")
    
    # Test sentence to indices
    test_sentence = "hello world"
    indices = vocab.sentence_to_indices(test_sentence, max_length=10)
    print(f"\nSentence: '{test_sentence}'")
    print(f"Indices: {indices}")
    
    # Test indices to sentence
    recovered = vocab.indices_to_sentence(indices)
    print(f"Recovered: '{recovered}'")
    
    print("✓ Vocabulary tests passed!")


if __name__ == '__main__':
    try:
        test_model_forward()
        test_vocabulary()
        print("\n" + "=" * 60)
        print("SUCCESS: All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

