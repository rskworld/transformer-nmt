"""
Inference Script for Transformer-based Neural Machine Translation

This script provides functionality for translating sentences using
a trained transformer model with beam search decoding.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
import torch.nn.functional as F
from transformer_model import Transformer
from data_preprocessing import load_vocabularies, normalize_string
import os
import argparse


class BeamSearchDecoder:
    """
    Beam search decoder for generating translations
    
    Author: Molla Samser (https://rskworld.in)
    """
    
    def __init__(self, model, tgt_vocab, device, beam_width=5, max_length=100):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.beam_width = beam_width
        self.max_length = max_length
    
    def translate(self, src_sentence, src_vocab):
        """
        Translate a source sentence using beam search
        
        Author: Molla Samser (https://rskworld.in)
        """
        self.model.eval()
        
        # Tokenize source sentence
        src_indices = src_vocab.sentence_to_indices(normalize_string(src_sentence))
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Create source mask
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        
        # Encode source
        with torch.no_grad():
            enc_output = self.model.encoder(src_tensor, src_mask)
        
        # Initialize beam
        beams = [([self.tgt_vocab.SOS_token], 0.0)]
        
        for _ in range(self.max_length):
            candidates = []
            
            for sequence, score in beams:
                # Check if sequence is complete
                if sequence[-1] == self.tgt_vocab.EOS_token:
                    candidates.append((sequence, score))
                    continue
                
                # Create target tensor
                tgt_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                
                # Create target mask
                tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(3)
                seq_length = len(sequence)
                nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
                nopeak_mask = nopeak_mask.to(self.device)
                tgt_mask = tgt_mask & nopeak_mask
                
                # Decode
                with torch.no_grad():
                    dec_output = self.model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
                    output = self.model.fc_out(dec_output)
                    log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                
                # Get top beam_width candidates
                top_probs, top_indices = torch.topk(log_probs, self.beam_width)
                
                for i in range(self.beam_width):
                    new_sequence = sequence + [top_indices[0][i].item()]
                    new_score = score + top_probs[0][i].item()
                    candidates.append((new_sequence, new_score))
            
            # Keep top beam_width sequences
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
            
            # Check if all beams are complete
            if all(beam[0][-1] == self.tgt_vocab.EOS_token for beam in beams):
                break
        
        # Return best sequence
        best_sequence = beams[0][0]
        return self.tgt_vocab.indices_to_sentence(best_sequence)


def greedy_decode(model, src_sentence, src_vocab, tgt_vocab, device, max_length=100):
    """
    Greedy decoding for translation (faster but less accurate than beam search)
    
    Author: Molla Samser (https://rskworld.in)
    """
    model.eval()
    
    # Tokenize source
    src_indices = src_vocab.sentence_to_indices(normalize_string(src_sentence))
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    
    # Create source mask
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
    
    # Encode
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    
    # Initialize target with SOS token
    tgt_sequence = [tgt_vocab.SOS_token]
    
    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_sequence], dtype=torch.long).to(device)
        
        # Create target mask
        tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(3)
        seq_length = len(tgt_sequence)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(device)
        tgt_mask = tgt_mask & nopeak_mask
        
        # Decode
        with torch.no_grad():
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            output = model.fc_out(dec_output)
            next_token = output[0, -1, :].argmax(dim=-1).item()
        
        tgt_sequence.append(next_token)
        
        if next_token == tgt_vocab.EOS_token:
            break
    
    return tgt_vocab.indices_to_sentence(tgt_sequence)


def load_model(model_path, device):
    """
    Load trained model from checkpoint
    
    Author: Molla Samser (https://rskworld.in)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = Transformer(
        src_vocab_size=checkpoint['src_vocab_size'],
        tgt_vocab_size=checkpoint['tgt_vocab_size'],
        d_model=checkpoint['d_model'],
        num_heads=checkpoint['num_heads'],
        num_encoder_layers=checkpoint['num_layers'],
        num_decoder_layers=checkpoint['num_layers'],
        d_ff=checkpoint['d_ff'],
        max_len=checkpoint['max_length'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def translate_sentence(sentence, model_path, vocab_dir, device=None, use_beam_search=True, beam_width=5):
    """
    Translate a single sentence
    
    Author: Molla Samser (https://rskworld.in)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabularies
    src_vocab, tgt_vocab = load_vocabularies(vocab_dir)
    
    # Load model
    model = load_model(model_path, device)
    
    # Translate
    if use_beam_search:
        decoder = BeamSearchDecoder(model, tgt_vocab, device, beam_width)
        translation = decoder.translate(sentence, src_vocab)
    else:
        translation = greedy_decode(model, sentence, src_vocab, tgt_vocab, device)
    
    return translation


def translate_file(input_file, output_file, model_path, vocab_dir, device=None, use_beam_search=True):
    """
    Translate sentences from a file
    
    Author: Molla Samser (https://rskworld.in)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabularies and model
    src_vocab, tgt_vocab = load_vocabularies(vocab_dir)
    model = load_model(model_path, device)
    
    if use_beam_search:
        decoder = BeamSearchDecoder(model, tgt_vocab, device)
    
    translations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                if use_beam_search:
                    translation = decoder.translate(sentence, src_vocab)
                else:
                    translation = greedy_decode(model, sentence, src_vocab, tgt_vocab, device)
                translations.append(translation)
    
    # Write translations
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
    
    print(f"Translated {len(translations)} sentences")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate using trained Transformer NMT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_dir', type=str, default='./models',
                       help='Directory containing vocabulary files')
    parser.add_argument('--sentence', type=str, default=None,
                       help='Single sentence to translate')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file with sentences to translate')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for translations')
    parser.add_argument('--use_beam_search', action='store_true',
                       help='Use beam search decoding (slower but better)')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Author: Molla Samser - https://rskworld.in")
    
    if args.sentence:
        translation = translate_sentence(
            args.sentence, args.model_path, args.vocab_dir,
            device, args.use_beam_search, args.beam_width
        )
        print(f"\nSource: {args.sentence}")
        print(f"Translation: {translation}")
    
    elif args.input_file and args.output_file:
        translate_file(
            args.input_file, args.output_file, args.model_path,
            args.vocab_dir, device, args.use_beam_search
        )
    
    else:
        print("Please provide either --sentence or both --input_file and --output_file")

