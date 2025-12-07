"""
REST API Server for Transformer NMT

Flask-based REST API server for serving translation requests.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import argparse

from inference import load_model, translate_sentence

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
model_path = None
vocab_dir = None
device = None


def init_model(model_path_arg, vocab_dir_arg='./models'):
    """
    Initialize the model for serving
    
    Author: Molla Samser (https://rskworld.in)
    """
    global model, model_path, vocab_dir, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = model_path_arg
    vocab_dir = vocab_dir_arg
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    print(f"Author: Molla Samser - https://rskworld.in")
    
    # Model will be loaded on first request
    print("Model will be loaded on first request...")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate endpoint
    
    Expected JSON:
    {
        "text": "sentence to translate",
        "use_beam_search": true,
        "beam_width": 5
    }
    
    Author: Molla Samser (https://rskworld.in)
    """
    global model
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        use_beam_search = data.get('use_beam_search', True)
        beam_width = data.get('beam_width', 5)
        
        # Load model on first request if not loaded
        if model is None:
            from inference import load_model, translate_sentence
            model = load_model(model_path, device)
        
        # Translate
        translation = translate_sentence(
            text,
            model_path,
            vocab_dir,
            device,
            use_beam_search,
            beam_width
        )
        
        return jsonify({
            'source': text,
            'translation': translation,
            'beam_search': use_beam_search
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/translate/batch', methods=['POST'])
def translate_batch():
    """
    Batch translate endpoint
    
    Expected JSON:
    {
        "texts": ["sentence1", "sentence2", ...],
        "use_beam_search": true,
        "beam_width": 5
    }
    
    Author: Molla Samser (https://rskworld.in)
    """
    global model
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field in request'}), 400
        
        texts = data['texts']
        use_beam_search = data.get('use_beam_search', True)
        beam_width = data.get('beam_width', 5)
        
        # Load model on first request if not loaded
        if model is None:
            model = load_model(model_path, device)
        
        # Translate all texts
        translations = []
        for text in texts:
            translation = translate_sentence(
                text,
                model_path,
                vocab_dir,
                device,
                use_beam_search,
                beam_width
            )
            translations.append(translation)
        
        return jsonify({
            'sources': texts,
            'translations': translations,
            'beam_search': use_beam_search
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start REST API server for Transformer NMT')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_dir', type=str, default='./models',
                       help='Directory containing vocabulary files')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    
    args = parser.parse_args()
    
    init_model(args.model_path, args.vocab_dir)
    
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  POST /translate - Single sentence translation")
    print(f"  POST /translate/batch - Batch translation")
    print(f"  GET /health - Health check")
    print(f"\nAuthor: Molla Samser - https://rskworld.in\n")
    
    app.run(host=args.host, port=args.port, debug=False)

