from flask import Flask, render_template, request
import sqlite3
import os
import threading
import webbrowser

from utils.predictor import predict_sleep_quality

app = Flask(__name__)

# =========================
# DATABASE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sleep.db")


# =========================
# INIT DB
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_duration REAL,
            bedtime_hour REAL,
            wake_hour REAL,
            caffeine_intake TEXT,
            exercise_duration REAL,
            screen_time REAL,
            mood TEXT,
            interruptions TEXT,
            stress_level REAL,
            result TEXT,
            score REAL
        )
    """)

    conn.commit()
    conn.close()


# =========================
# TIME CONVERTER FIX 🔥
# =========================
def time_to_float(t):
    try:
        h, m = t.split(":")
        return float(h) + float(m) / 60
    except:
        return 0.0


# =========================
# HOME
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    # 🔥 FIX: convert time fields properly
    data["bedtime_hour"] = time_to_float(data.get("bedtime_hour", "23:00"))
    data["wake_hour"] = time_to_float(data.get("wake_hour", "07:00"))

    # ML prediction
    result = predict_sleep_quality(data)

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            sleep_duration, bedtime_hour, wake_hour,
            caffeine_intake, exercise_duration,
            screen_time, mood, interruptions,
            stress_level, result, score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("sleep_duration"),
        data.get("bedtime_hour"),
        data.get("wake_hour"),
        data.get("caffeine_intake"),
        data.get("exercise_duration"),
        data.get("screen_time_before_bed"),
        data.get("mood_before_sleep"),
        data.get("sleep_interruptions"),
        data.get("stress_level"),
        result["quality"],
        result["sleep_score"]
    ))

    conn.commit()
    conn.close()

    return render_template("result.html", result=result)


# =========================
# AUTO BROWSER (SAFE)
# =========================
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    init_db()

    threading.Timer(1.2, open_browser).start()

    app.run(debug=True, use_reloader=False)