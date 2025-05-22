#!/bin/bash
# Reduce memory usage
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs

# Django setup
python manage.py migrate
python manage.py collectstatic --noinput

# Start Gunicorn with minimal resources
gunicorn admissionchatbot.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 300 \
    --preload  # Reduces memory duplication
