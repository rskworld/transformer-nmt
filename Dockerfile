# Dockerfile for Transformer-based Neural Machine Translation
# Author: Molla Samser
# Website: https://rskworld.in
# Email: help@rskworld.in, support@rskworld.in
# Phone: +91 93305 39277

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p models data

# Expose port for API server
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "api_server.py", "--model_path", "models/best_model.pt", "--port", "5000"]

