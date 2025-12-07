"""
Data Preprocessing Utilities for Transformer NMT

This module contains utilities for preprocessing parallel corpora,
tokenization, vocabulary building, and data loading for training.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os


class Vocabulary:
    """
    Vocabulary builder for source and target languages
    
    Creates word-to-index and index-to-word mappings with special tokens
    for padding, unknown words, start of sequence, and end of sequence.
    
    Author: Molla Samser (https://rskworld.in)
    """
    
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
        self.add_word('<PAD>')
        self.add_word('<SOS>')
        self.add_word('<EOS>')
        self.add_word('<UNK>')
    
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.word_count[word] += 1
    
    def build_vocabulary(self, sentences, min_freq=2):
        """
        Build vocabulary from sentences with minimum frequency threshold
        
        Args:
            sentences: List of sentences
            min_freq: Minimum frequency for a word to be included
        """
        for sentence in sentences:
            self.add_sentence(sentence)
        
        # Add words that meet frequency threshold
        for word, count in self.word_count.items():
            if count >= min_freq:
                self.add_word(word)
    
    def sentence_to_indices(self, sentence, max_length=None):
        """
        Convert sentence to list of indices
        
        Args:
            sentence: Input sentence string
            max_length: Maximum length (padding/truncation)
        """
        indices = [self.SOS_token]
        words = sentence.split()
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.UNK_token)
        
        indices.append(self.EOS_token)
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
                indices[-1] = self.EOS_token
            else:
                indices.extend([self.PAD_token] * (max_length - len(indices)))
        
        return indices
    
    def indices_to_sentence(self, indices):
        """
        Convert list of indices back to sentence
        
        Args:
            indices: List of word indices
        """
        sentence = []
        for idx in indices:
            if idx == self.EOS_token:
                break
            if idx not in [self.PAD_token, self.SOS_token]:
                word = self.idx2word.get(idx, '<UNK>')
                sentence.append(word)
        return ' '.join(sentence)
    
    def __len__(self):
        return len(self.word2idx)


def normalize_string(s):
    """
    Normalize string: lowercase, trim, remove non-letter characters
    
    Author: Molla Samser (https://rskworld.in)
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


class TranslationDataset(Dataset):
    """
    Dataset class for parallel translation pairs
    
    Author: Molla Samser (https://rskworld.in)
    """
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_indices = self.src_vocab.sentence_to_indices(src_sentence, self.max_length)
        tgt_indices = self.tgt_vocab.sentence_to_indices(tgt_sentence, self.max_length)
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def load_data(file_path, num_pairs=None):
    """
    Load parallel corpus from file
    
    Expected format: source_sentence ||| target_sentence
    
    Author: Molla Samser (https://rskworld.in)
    """
    src_sentences = []
    tgt_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_pairs and i >= num_pairs:
                break
            
            if '|||' in line:
                parts = line.strip().split('|||')
                if len(parts) == 2:
                    src_sentences.append(normalize_string(parts[0]))
                    tgt_sentences.append(normalize_string(parts[1]))
    
    return src_sentences, tgt_sentences


def create_data_loader(src_sentences, tgt_sentences, src_vocab, tgt_vocab, 
                       batch_size=32, max_length=100, shuffle=True):
    """
    Create DataLoader for training
    
    Author: Molla Samser (https://rskworld.in)
    """
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, 
                                 tgt_vocab, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_vocabularies(src_vocab, tgt_vocab, save_dir='./models'):
    """
    Save vocabulary objects to disk
    
    Author: Molla Samser (https://rskworld.in)
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'src_vocab.pkl'), 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(os.path.join(save_dir, 'tgt_vocab.pkl'), 'wb') as f:
        pickle.dump(tgt_vocab, f)


def load_vocabularies(load_dir='./models'):
    """
    Load vocabulary objects from disk
    
    Author: Molla Samser (https://rskworld.in)
    """
    with open(os.path.join(load_dir, 'src_vocab.pkl'), 'rb') as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(load_dir, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_vocab = pickle.load(f)
    return src_vocab, tgt_vocab


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    
    Author: Molla Samser (https://rskworld.in)
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch

