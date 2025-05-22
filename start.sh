#!/bin/bash
python manage.py migrate
python manage.py collectstatic --noinput
gunicorn admissionchatbot.wsgi:application --bind 0.0.0.0:$PORT