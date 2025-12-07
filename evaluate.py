"""
Model Evaluation Script with BLEU Score

This script evaluates the trained transformer model using BLEU scores
and other translation quality metrics.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
import argparse
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

from inference import load_model, greedy_decode, BeamSearchDecoder
from data_preprocessing import load_vocabularies, normalize_string, load_data

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score for a single sentence pair
    
    Author: Molla Samser (https://rskworld.in)
    """
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)


def evaluate_model(
    model_path,
    test_data_path,
    vocab_dir='./models',
    use_beam_search=True,
    beam_width=5,
    device=None
):
    """
    Evaluate model on test data
    
    Author: Molla Samser (https://rskworld.in)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Author: Molla Samser - https://rskworld.in")
    
    # Load vocabularies and model
    print("Loading vocabularies...")
    src_vocab, tgt_vocab = load_vocabularies(vocab_dir)
    
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load test data
    print("Loading test data...")
    src_sentences, tgt_sentences = load_data(test_data_path)
    
    print(f"Evaluating on {len(src_sentences)} sentence pairs...")
    
    # Translate all sentences
    translations = []
    bleu_scores = []
    
    if use_beam_search:
        decoder = BeamSearchDecoder(model, tgt_vocab, device, beam_width)
        for i, src_sentence in enumerate(src_sentences):
            translation = decoder.translate(src_sentence, src_vocab)
            translations.append(translation)
            
            # Calculate BLEU score
            if i < len(tgt_sentences):
                bleu = calculate_bleu(tgt_sentences[i], translation)
                bleu_scores.append(bleu)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(src_sentences)} sentences")
    else:
        for i, src_sentence in enumerate(src_sentences):
            translation = greedy_decode(model, src_sentence, src_vocab, tgt_vocab, device)
            translations.append(translation)
            
            if i < len(tgt_sentences):
                bleu = calculate_bleu(tgt_sentences[i], translation)
                bleu_scores.append(bleu)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(src_sentences)} sentences")
    
    # Calculate statistics
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Total sentences: {len(src_sentences)}")
        print(f"Average BLEU score: {avg_bleu:.4f}")
        print(f"Max BLEU score: {max(bleu_scores):.4f}")
        print(f"Min BLEU score: {min(bleu_scores):.4f}")
        print(f"{'='*60}\n")
        
        # Show some examples
        print("Sample Translations:")
        print("-" * 60)
        for i in range(min(5, len(src_sentences))):
            print(f"\nExample {i+1}:")
            print(f"Source: {src_sentences[i]}")
            if i < len(tgt_sentences):
                print(f"Reference: {tgt_sentences[i]}")
            print(f"Translation: {translations[i]}")
            if i < len(bleu_scores):
                print(f"BLEU Score: {bleu_scores[i]:.4f}")
        
        return avg_bleu, bleu_scores, translations
    
    return None, None, translations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer NMT Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--vocab_dir', type=str, default='./models',
                       help='Directory containing vocabulary files')
    parser.add_argument('--use_beam_search', action='store_true',
                       help='Use beam search decoding')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search')
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model_path,
        args.test_data,
        args.vocab_dir,
        args.use_beam_search,
        args.beam_width
    )

