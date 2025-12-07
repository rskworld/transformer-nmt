# Quick Start Guide

**Author:** Molla Samser  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277

## Installation

1. **Install Python 3.8+** (if not already installed)

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python test_model.py
```

## Quick Training Example

1. **Prepare your data** in format: `source ||| target` (see `example_data.txt`)

2. **Train the model:**
```bash
python train.py --data_path example_data.txt --num_epochs 10 --batch_size 16
```

3. **Translate sentences:**
```bash
python inference.py --model_path models/best_model.pt --sentence "hello world"
```

## Using the Jupyter Notebook

```bash
jupyter notebook transformer_nmt_demo.ipynb
```

## Basic Usage

### Training
```bash
python train.py \
    --data_path your_data.txt \
    --num_epochs 50 \
    --batch_size 32 \
    --save_dir ./models
```

### Translation
```bash
# Single sentence
python inference.py \
    --model_path models/best_model.pt \
    --sentence "Your sentence here"

# Batch translation
python inference.py \
    --model_path models/best_model.pt \
    --input_file input.txt \
    --output_file output.txt
```

## Project Structure

- `transformer_model.py` - Core transformer architecture
- `data_preprocessing.py` - Data utilities
- `train.py` - Training script
- `inference.py` - Translation script
- `transformer_nmt_demo.ipynb` - Jupyter notebook
- `test_model.py` - Model testing
- `utils.py` - Utility functions
- `config.py` - Configuration settings

For detailed documentation, see `README.md`

**Contact:** help@rskworld.in | https://rskworld.in

