FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng || true

COPY . .

EXPOSE 5000

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000}