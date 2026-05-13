#!/bin/bash

# Start Flask backend in background
python app.py &

# Start Streamlit frontend
streamlit run streamlit_app.py \
  --server.port=${PORT:-8501} \
  --server.address=0.0.0.0