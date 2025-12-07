# Transformer-based Neural Machine Translation

**Author:** Molla Samser  
**Website:** [https://rskworld.in](https://rskworld.in)  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277  
**Designer & Tester:** Rima Khatun

A complete implementation of Transformer architecture for neural machine translation with self-attention and multi-head attention mechanisms for high-quality translation generation.

## Features

- ✅ **Transformer Architecture**: Complete encoder-decoder transformer implementation
- ✅ **Self-Attention Mechanism**: Allows words to attend to all other words in the sequence
- ✅ **Multi-Head Attention**: Multiple attention heads capture different types of relationships
- ✅ **Positional Encoding**: Adds positional information to word embeddings
- ✅ **Beam Search Decoding**: Generates high-quality translations
- ✅ **Easy-to-use Training Script**: Simple command-line interface for training
- ✅ **Flexible Inference**: Support for both greedy and beam search decoding

## Project Structure

```
transformer-nmt/
├── transformer_model.py      # Core transformer architecture
├── data_preprocessing.py     # Data loading and vocabulary building
├── train.py                  # Training script with full pipeline
├── inference.py              # Inference and translation utilities
├── transformer_nmt_demo.ipynb # Jupyter notebook demonstration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # Saved models and vocabularies (created during training)
```

## Installation

1. Clone the repository or download the source code

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have PyTorch installed (CPU or CUDA version):

```bash
# For CPU
pip install torch

# For CUDA (GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Prepare Your Data

Prepare a parallel corpus file with format:
```
source_sentence_1 ||| target_sentence_1
source_sentence_2 ||| target_sentence_2
...
```

Example:
```
hello world ||| bonjour le monde
how are you ||| comment allez-vous
```

### 2. Train the Model

```bash
python train.py \
    --data_path data/parallel_corpus.txt \
    --num_epochs 50 \
    --batch_size 32 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --lr 0.0001 \
    --max_length 100 \
    --save_dir ./models
```

**Training Parameters:**
- `--data_path`: Path to your parallel corpus file
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of encoder/decoder layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--dropout`: Dropout rate (default: 0.1)
- `--lr`: Learning rate (default: 0.0001)
- `--max_length`: Maximum sequence length (default: 100)
- `--min_freq`: Minimum word frequency for vocabulary (default: 2)
- `--num_pairs`: Number of sentence pairs to use (optional, uses all if not specified)
- `--save_dir`: Directory to save models (default: ./models)

### 3. Translate Sentences

**Single Sentence Translation:**
```bash
python inference.py \
    --model_path models/best_model.pt \
    --sentence "Hello, how are you?" \
    --use_beam_search \
    --beam_width 5
```

**Batch Translation from File:**
```bash
python inference.py \
    --model_path models/best_model.pt \
    --input_file input_sentences.txt \
    --output_file translations.txt \
    --use_beam_search \
    --beam_width 5
```

**Inference Parameters:**
- `--model_path`: Path to trained model checkpoint
- `--vocab_dir`: Directory containing vocabulary files (default: ./models)
- `--sentence`: Single sentence to translate
- `--input_file`: Input file with sentences (one per line)
- `--output_file`: Output file for translations
- `--use_beam_search`: Use beam search (better quality, slower)
- `--beam_width`: Number of beams for beam search (default: 5)

## Architecture Details

### Transformer Components

1. **Encoder**: 
   - Stack of identical encoder layers
   - Each layer has multi-head self-attention and feed-forward network
   - Residual connections and layer normalization

2. **Decoder**:
   - Stack of identical decoder layers
   - Each layer has masked self-attention, encoder-decoder attention, and feed-forward network
   - Residual connections and layer normalization

3. **Attention Mechanisms**:
   - Self-attention in encoder
   - Masked self-attention in decoder
   - Encoder-decoder attention in decoder

4. **Positional Encoding**:
   - Sinusoidal encoding added to embeddings
   - Allows model to understand word order

### Model Hyperparameters

Default configuration:
- Model dimension: 512
- Number of heads: 8
- Number of layers: 6 (encoder and decoder)
- Feed-forward dimension: 2048
- Dropout: 0.1
- Maximum sequence length: 100

## Jupyter Notebook

The project includes a comprehensive Jupyter notebook (`transformer_nmt_demo.ipynb`) that demonstrates:
- Model architecture visualization
- Self-attention mechanism explanation
- Training setup examples
- Translation inference examples
- Positional encoding visualization

To run the notebook:
```bash
jupyter notebook transformer_nmt_demo.ipynb
```

## Example Workflow

1. **Prepare Data**: Create a parallel corpus file
2. **Train Model**: Run training script with your data
3. **Evaluate**: Check validation loss during training
4. **Translate**: Use inference script for translations
5. **Fine-tune**: Adjust hyperparameters if needed

## Technical Details

### Self-Attention Formula

The attention mechanism computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Dimension of keys

### Multi-Head Attention

Multiple attention heads run in parallel:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

Each head captures different types of relationships.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm (for progress bars)
- matplotlib (for visualizations)
- Jupyter (for notebook)

See `requirements.txt` for complete list.

## License

This project is provided for educational purposes. For more information, visit [rskworld.in](https://rskworld.in).

## Contact

**Author:** Molla Samser  
**Website:** [https://rskworld.in](https://rskworld.in)  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277  
**Designer & Tester:** Rima Khatun

## Acknowledgments

This implementation is based on the "Attention Is All You Need" paper by Vaswani et al. (2017).

For more programming resources, source code, and development tools, visit [rskworld.in](https://rskworld.in).

