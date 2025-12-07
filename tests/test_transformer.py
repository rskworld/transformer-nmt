"""
Unit Tests for Transformer NMT Model

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_model import (
    Transformer, Encoder, Decoder, MultiHeadAttention,
    PositionalEncoding, EncoderLayer, DecoderLayer
)
from data_preprocessing import Vocabulary


class TestTransformerComponents(unittest.TestCase):
    """Test transformer model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.src_len = 10
        self.tgt_len = 12
        self.d_model = 128
        self.num_heads = 4
        self.d_ff = 512
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
    
    def test_positional_encoding(self):
        """Test positional encoding"""
        pe = PositionalEncoding(self.d_model, max_len=100, dropout=0)
        x = torch.randn(10, self.batch_size, self.d_model)
        output = pe(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_multi_head_attention(self):
        """Test multi-head attention"""
        attn = MultiHeadAttention(self.d_model, self.num_heads)
        x = torch.randn(self.batch_size, self.src_len, self.d_model)
        
        output, weights = attn(x, x, x)
        
        self.assertEqual(output.shape, (self.batch_size, self.src_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.src_len, self.src_len))
    
    def test_encoder_layer(self):
        """Test encoder layer"""
        layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff)
        x = torch.randn(self.batch_size, self.src_len, self.d_model)
        mask = torch.ones(self.batch_size, 1, 1, self.src_len).bool()
        
        output = layer(x, mask)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_decoder_layer(self):
        """Test decoder layer"""
        layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff)
        x = torch.randn(self.batch_size, self.tgt_len, self.d_model)
        enc_output = torch.randn(self.batch_size, self.src_len, self.d_model)
        src_mask = torch.ones(self.batch_size, 1, 1, self.src_len).bool()
        tgt_mask = torch.ones(self.batch_size, 1, self.tgt_len, self.tgt_len).bool()
        
        output = layer(x, enc_output, src_mask, tgt_mask)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_transformer_model(self):
        """Test complete transformer model"""
        model = Transformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=self.d_ff,
            max_len=100,
            dropout=0.1
        )
        
        src = torch.randint(1, self.src_vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(1, self.tgt_vocab_size, (self.batch_size, self.tgt_len))
        
        output = model(src, tgt[:, :-1])
        
        self.assertEqual(output.shape, (self.batch_size, self.tgt_len - 1, self.tgt_vocab_size))


class TestVocabulary(unittest.TestCase):
    """Test vocabulary building"""
    
    def test_vocabulary_building(self):
        """Test vocabulary creation"""
        vocab = Vocabulary('test')
        sentences = [
            "hello world",
            "hello python",
            "world peace"
        ]
        
        vocab.build_vocabulary(sentences, min_freq=1)
        
        self.assertGreater(len(vocab), 0)
        self.assertEqual(vocab.PAD_token, 0)
        self.assertEqual(vocab.SOS_token, 1)
        self.assertEqual(vocab.EOS_token, 2)
        self.assertEqual(vocab.UNK_token, 3)
    
    def test_sentence_to_indices(self):
        """Test sentence to indices conversion"""
        vocab = Vocabulary('test')
        vocab.build_vocabulary(["hello world"], min_freq=1)
        
        indices = vocab.sentence_to_indices("hello world", max_length=10)
        
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices[0], vocab.SOS_token)
        self.assertEqual(indices[-1], vocab.PAD_token or vocab.EOS_token)


if __name__ == '__main__':
    unittest.main()

