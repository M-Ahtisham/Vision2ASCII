#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start HTTP server in a subshell
(
    cd src/html || exit 1
    python3 -m http.server 8000
) &

# Run Streamlit app
streamlit run main.py