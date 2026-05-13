FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download NLTK data if your model/preprocessing needs it
RUN python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng || true

# Copy project files
COPY . .

# Give permission to start script
RUN chmod +x start.sh

# Streamlit port
EXPOSE 8501

# Flask internal port
EXPOSE 5000

# Health check for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Flask backend + Streamlit frontend
CMD ["./start.sh"]