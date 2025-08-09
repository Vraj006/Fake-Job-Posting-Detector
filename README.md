# Fake Job Detector

ML-powered web app to detect fake job postings using Flask API + React frontend.

## Setup

### Backend
```bash
cd fake-job-detector-api
pip install -r requirements.txt
python app.py
```
API runs on `http://localhost:5000`

### Frontend
```bash
cd fake-job-detector-frontend
npm install
npm start
```
App runs on `http://localhost:3000`

## API Usage

**POST** `/predict`
```json
{"job_description": "Your job text here"}
```

**Response:**
```json
{"prediction": "Real/Fake", "confidence": 0.85}
```

## Deployment

### Render (Backend)
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`

### Frontend
- Update `.env.production` with API URL
- Run `npm run build`
- Deploy `build` folder

## Tech Stack
- **Backend:** Flask, scikit-learn, pandas
- **Frontend:** React, Lucide icons
- **ML Model:** `fake_job_detector_v1.pkl`
