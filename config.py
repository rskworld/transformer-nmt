"""
Configuration file for Transformer NMT

This file contains default configuration parameters for training and inference.
Modify these values according to your requirements.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

# Model Architecture Parameters
D_MODEL = 512          # Model dimension
NUM_HEADS = 8          # Number of attention heads
NUM_LAYERS = 6         # Number of encoder/decoder layers
D_FF = 2048            # Feed-forward network dimension
DROPOUT = 0.1          # Dropout rate
MAX_LEN = 5000         # Maximum sequence length for positional encoding

# Training Parameters
BATCH_SIZE = 32        # Batch size
NUM_EPOCHS = 50        # Number of training epochs
LEARNING_RATE = 0.0001 # Learning rate
MIN_FREQ = 2           # Minimum word frequency for vocabulary
MAX_LENGTH = 100       # Maximum sequence length for training
GRAD_CLIP = 1.0        # Gradient clipping value

# Data Parameters
TRAIN_SPLIT = 0.9      # Train/validation split ratio

# Inference Parameters
BEAM_WIDTH = 5         # Beam width for beam search
USE_BEAM_SEARCH = True # Whether to use beam search or greedy decoding
MAX_DECODE_LENGTH = 100 # Maximum decoding length

# Paths
DEFAULT_MODEL_DIR = './models'
DEFAULT_DATA_DIR = './data'

# Special Tokens (defined in Vocabulary class)
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

