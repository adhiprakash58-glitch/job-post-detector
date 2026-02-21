# HireShield 360

HireShield 360 is a Flask web app that detects potentially fraudulent job and internship posts from LinkedIn, Instagram, and email-style content.

## Features
- Paste text content or upload screenshot images of job posts.
- OCR extraction from uploaded images (via `pytesseract`) when available.
- NLP pipeline using **TF-IDF** + **Logistic Regression** to classify posts as **Genuine** or **Fraudulent**.
- Confidence score returned in real time.
- Domain verification using extracted recruiter emails:
  - flags suspicious free email providers,
  - checks mismatch against inferred official domains from company names in content.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## Notes
- OCR requires a local Tesseract binary. If missing, the app still works for pasted text and shows an OCR availability message.
- The model is trained at startup on a small built-in labeled dataset for demonstration.
