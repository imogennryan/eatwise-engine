#!/bin/bash
# Wrapper so the Preview tool can inject PORT via environment variable
STREAMLIT_SERVER_PORT=${PORT:-8503} /opt/anaconda3/bin/streamlit run app.py --server.headless true
