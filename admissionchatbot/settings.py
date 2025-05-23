"""
Django settings for admissionchatbot project.
"""

import os
import nltk
import dj_database_url
from django.core.management.utils import get_random_secret_key

# Build paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))

# Security Settings (Critical for Production)
SECRET_KEY = os.environ.get('SECRET_KEY', get_random_secret_key())
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1,dn-college-inquiry-bot.onrender.com').split(',')
CSRF_TRUSTED_ORIGINS = ['https://dn-college-inquiry-bot.onrender.com']
CORS_ALLOWED_ORIGINS = ['https://dn-college-inquiry-bot.onrender.com']

# NLTK Configuration
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Application Definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'cgpit',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Database Configuration (Use Render PostgreSQL)
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3'),
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# Static Files Configuration (Critical for Render)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'admissionchatbot/static')]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Security Headers
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_HSTS_SECONDS = 31536000  # 1 year

# Model Configuration
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# TensorFlow Memory Optimization
if not DEBUG:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.backend.clear_session()

# Rest of the configuration remains the same...
