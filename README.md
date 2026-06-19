# 🌙 SleepSense AI — Sleep Quality Predictor

> AI-Powered Sleep Quality Prediction and Personalised Wellness Recommendation System

## 📋 Overview

A complete Machine Learning web application that analyses lifestyle, behavioural, and physiological factors to predict sleep quality (Good / Average / Poor), generates a Sleep Score (0–100), and provides personalised wellness recommendations.

## 🏗️ Project Structure

```
sleep_predictor/
├── app.py                      ← Flask web application (main entry point)
├── generate_and_train.py       ← ML pipeline: dataset, training, visualisations
├── utils/
│   └── predictor.py            ← Prediction engine & recommendation generator
├── models/
│   ├── sleep_model.pkl         ← Best trained model (SVM, ~81.8% accuracy)
│   ├── scaler.pkl              ← Fitted StandardScaler
│   └── meta.json               ← Model metadata & feature importance
├── data/
│   ├── sleep_data.csv          ← Generated training dataset (2000 records)
│   └── sleep_history.db        ← SQLite database for user predictions
├── static/
│   └── images/                 ← EDA & model performance charts (PNG)
└── templates/
    ├── base.html               ← Base layout with navbar
    ├── index.html              ← Home page & prediction form
    ├── result.html             ← Prediction results dashboard
    ├── dashboard.html          ← Analytics dashboard (Chart.js)
    ├── history.html            ← Prediction history table
    └── analytics.html          ← ML/EDA visualisations
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Train the model (already done — skip if models/ exists)
```bash
python generate_and_train.py
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

## 🤖 ML Models Trained

| Model                | Test Accuracy | CV Mean  |
|---------------------|--------------|---------|
| Logistic Regression  | 73.3%        | 70.9%   |
| Decision Tree        | 71.3%        | 66.7%   |
| Random Forest        | 79.5%        | 76.5%   |
| **SVM ✅ (best)**   | **81.8%**    | **77.9%** |

## 📥 Input Features

- Sleep Duration (hours)
- Bedtime & Wake-up Time
- Caffeine Intake (None / Low / Moderate / High)
- Exercise Duration (minutes)
- Screen Time Before Bed (minutes)
- Stress Level (0–10 slider)
- Mood Before Sleep (Happy / Neutral / Sad / Anxious)
- Sleep Interruptions (Yes / No)

## 📤 Output

- **Sleep Quality**: Good / Average / Poor
- **Sleep Score**: 0–100
- **Confidence**: Percentage from model probability
- **Class Probabilities**: Per-class breakdown
- **Positive Factors**: What's helping your sleep
- **Negative Factors**: What's hurting your sleep
- **Personalised Recommendations**: Up to 6 specific, evidence-based tips

## 🗄️ Database

SQLite database (`data/sleep_history.db`) stores all predictions with:
- Date/time, all input features
- Sleep score, quality, confidence
- Serialised recommendations

View history at `/history` and analytics at `/dashboard`.

## 🎨 Tech Stack

- **ML**: scikit-learn (LR, DT, RF, SVM), joblib
- **Data**: pandas, numpy
- **Visualisations**: matplotlib, seaborn, Chart.js
- **Backend**: Flask, SQLite
- **Frontend**: HTML5, CSS3, Poppins + Roboto fonts
- **Charts**: Chart.js 4.4

## 📝 API Endpoint

`POST /api/predict` — JSON input, JSON output:
```json
{
  "sleep_duration": 5,
  "bedtime_hour": 1.0,
  "wake_hour": 6.5,
  "caffeine_intake": "High",
  "exercise_duration": 0,
  "screen_time_before_bed": 180,
  "stress_level": 9,
  "mood_before_sleep": "Anxious",
  "sleep_interruptions": "Yes"
}
```
