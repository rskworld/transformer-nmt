# REST API Documentation

**Author:** Molla Samser  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277

## Getting Started

Start the API server:

```bash
python api_server.py --model_path models/best_model.pt --port 5000
```

The server will be available at `http://localhost:5000`

## Endpoints

### Health Check

**GET** `/health`

Check if the server is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Single Sentence Translation

**POST** `/translate`

Translate a single sentence.

**Request Body:**
```json
{
  "text": "Hello, how are you?",
  "use_beam_search": true,
  "beam_width": 5
}
```

**Response:**
```json
{
  "source": "Hello, how are you?",
  "translation": "Bonjour, comment allez-vous?",
  "beam_search": true
}
```

**Parameters:**
- `text` (required): Sentence to translate
- `use_beam_search` (optional): Use beam search decoding (default: true)
- `beam_width` (optional): Beam width for beam search (default: 5)

### Batch Translation

**POST** `/translate/batch`

Translate multiple sentences at once.

**Request Body:**
```json
{
  "texts": [
    "Hello, how are you?",
    "What is your name?",
    "Thank you very much"
  ],
  "use_beam_search": true,
  "beam_width": 5
}
```

**Response:**
```json
{
  "sources": [
    "Hello, how are you?",
    "What is your name?",
    "Thank you very much"
  ],
  "translations": [
    "Bonjour, comment allez-vous?",
    "Quel est votre nom?",
    "Merci beaucoup"
  ],
  "beam_search": true
}
```

## Example Usage

### Using cURL

```bash
# Single translation
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'

# Batch translation
curl -X POST http://localhost:5000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "Goodbye"]}'
```

### Using Python

```python
import requests

# Single translation
response = requests.post(
    'http://localhost:5000/translate',
    json={'text': 'Hello, how are you?'}
)
result = response.json()
print(result['translation'])

# Batch translation
response = requests.post(
    'http://localhost:5000/translate/batch',
    json={'texts': ['Hello', 'Goodbye']}
)
translations = response.json()['translations']
```

### Using JavaScript

```javascript
// Single translation
fetch('http://localhost:5000/translate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Hello, how are you?'})
})
.then(res => res.json())
.then(data => console.log(data.translation));
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Successful translation
- `400 Bad Request`: Invalid request (missing fields, etc.)
- `500 Internal Server Error`: Server error (model loading, etc.)

Error response format:
```json
{
  "error": "Error message here"
}
```

## CORS

CORS is enabled by default, allowing requests from any origin. Modify `api_server.py` to restrict origins if needed.

## Contact

For issues or questions, contact:
- **Email:** help@rskworld.in
- **Website:** https://rskworld.in

