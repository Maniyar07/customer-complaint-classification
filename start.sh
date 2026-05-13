#!/bin/sh

echo "Starting Flask backend on port 5000..."
gunicorn app:app --bind 127.0.0.1:5000 &

echo "Starting Streamlit frontend on Render port..."
streamlit run streamlit_app.py \
  --server.port=${PORT:-8501} \
  --server.address=0.0.0.0