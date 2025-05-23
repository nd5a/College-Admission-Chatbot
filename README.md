# College-Admission-Chatbot

[![Render Deployment](https://img.shields.io/badge/Deployed%20on-Render-%2300C7B7)](https://dn-college-inquiry-bot.onrender.com)

An AI-powered chatbot built with Django and NLP to handle college admission queries, featuring intent classification, dynamic responses, and real-time web scraping.

---
![Screenshot 2025-05-23 194601](https://github.com/user-attachments/assets/24a858f2-4b24-4777-9c56-ce8a2545f241)
---
![Screenshot 2025-05-23 194552](https://github.com/user-attachments/assets/912cb399-8cda-4d65-ac1d-1351dbe9dd21)
---
![Screenshot 2025-05-23 194541](https://github.com/user-attachments/assets/cf7ab818-ece0-4489-bb1f-81748b16ab6d)
---

## Features ✨
- **NLP Processing**: Intent detection using NLTK and TextBlob
- **Machine Learning**: Custom neural network with TensorFlow/Keras (85% accuracy)
- **Web Integration**: Real-time data scraping with BeautifulSoup
- **Security**: CSRF/CORS protection, HTTPS enforcement
- **Database**: SQLite/PostgreSQL conversation logging



## Installation & Setup 🛠️

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/college-admission-chatbot.git
cd college-admission-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader -d ./nltk_data punkt wordnet
```

### 4. Configure Environment
Create `.env` file:
```ini
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-render-domain.onrender.com
```

### 5. Database Setup
```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

---

## Configuration ⚙️

### `settings.py` Essentials
```python
# Security
CSRF_TRUSTED_ORIGINS = ['https://your-domain.onrender.com']
CORS_ALLOWED_ORIGINS = ['https://your-domain.onrender.com']

# Static Files
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
```

---

## Deployment on Render 🚀

1. **New Web Service**  
   Connect your GitHub repository

2. **Environment Setup**  
   ```ini
   PYTHON_VERSION = 3.11.4
   BUILD_COMMAND = pip install -r requirements.txt && python -m nltk.downloader -d ./nltk_data punkt wordnet
   START_COMMAND = python manage.py migrate && python manage.py collectstatic --noinput && gunicorn admissionchatbot.wsgi:application
   ```

3. **Add PostgreSQL**  
   Create database and add DATABASE_URL to environment variables

4. **Domain Setup**  
   Add custom domain in Render dashboard

---

## Usage 💬

### Local Development
```bash
python manage.py runserver
```
Visit `http://localhost:8000`

### Production Features
- HTTPS enforcement
- CSRF protection
- Gzip compression
- Static file caching

---

## Troubleshooting 🔧

| Error | Solution |
|-------|----------|
| `NLTK data not found` | Run `python -m nltk.downloader -d ./nltk_data punkt wordnet` |
| `CSRF verification failed` | Verify `CSRF_TRUSTED_ORIGINS` in settings |
| `TensorFlow AVX warnings` | Add `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` to settings |

---

## License 📄
MIT License - See [LICENSE](LICENSE)

---

**Note:** Replace all `your-domain` references with your actual domain name before deployment.
