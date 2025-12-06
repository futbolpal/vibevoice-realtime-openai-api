#!/bin/bash

# Ensure models directory is writable
if [ -d "/home/ubuntu/app/models" ]; then
    sudo chown -R ubuntu:ubuntu /home/ubuntu/app/models
fi

# Run the app
exec .venv/bin/python vibevoice_realtime_openai_api.py --port 8880
