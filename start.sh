#!/usr/bin/env bash
set -e

# 1. Cài spaCy model
python -m spacy download en_core_web_sm

# 2. Khởi động Gunicorn
exec gunicorn app:app \
  --bind 0.0.0.0:$PORT \
  --workers 2 \
  --timeout 120
