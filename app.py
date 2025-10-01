from flask import Flask, request, jsonify
import joblib
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import gdown  
from model import load_model_and_encoders
from datetime import datetime

app = Flask(__name__)

# ---------- Config ----------
MODEL_PATH = "model/pujPassModel.pkl"
ENCODER_PATH = "model/encoders.pkl"

# Google Drive File IDs
MODEL_ID = "1mXS5wishkGqjstxVdKUadkpDuKg6TQ6G"
ENCODER_ID = "1VhQw6AQWZ-_vRF_f9piD4ELsahsyeAWO"
# ----------------------------

def download_from_drive(file_id, dest_path):
    """Download a file from Google Drive if it does not already exist."""
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {dest_path} from Google Drive...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest_path, quiet=False)
        print(f"✅ {dest_path} downloaded successfully!")

# Ensure model + encoder exist
download_from_drive(MODEL_ID, MODEL_PATH)
download_from_drive(ENCODER_ID, ENCODER_PATH)

# Load model + encoder + stops
model, encoder = load_model_and_encoders(MODEL_PATH, ENCODER_PATH)
stops_df = pd.read_csv("data/updatedDataset.csv")[["Stop", "Latitude", "Longitude"]].drop_duplicates()


@app.route("/")
def home():
    return "🚍 Jeepney Passenger Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        stop = data["Stop"]
        day = data["DayOfWeek"]
        hour = int(data["Hour"])
        season = data["Season"]
        event = data["Event"]

        # Encode inputs
        X_pred = pd.DataFrame([[stop, day, hour, season, event]],
                              columns=["Stop", "DayOfWeek", "Hour", "Season", "Event"])
        X_pred[["Stop", "DayOfWeek", "Season", "Event"]] = encoder.transform(
            X_pred[["Stop", "DayOfWeek", "Season", "Event"]]
        )

        # Predict passenger volume
        vol = model.predict(X_pred)[0]

        return jsonify({
            "Stop": stop,
            "DayOfWeek": day,
            "Hour": hour,
            "Season": season,
            "Event": event,
            "PredictedPassengerVolume": int(vol)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

def get_current_season_and_event():
    now = datetime.now()
    month = now.month
    day = now.day

    # Default values
    season = "Regular"
    event = "Regular"

    if month == 12:
        season = "Christmas"
    elif month in [6, 7]:
        season = "Summer"
    elif month in [3, 4]:
        season = "Graduation"
    elif month == 8:
        season = "Kadayawan"


    if (month == 12 and day == 31) or (month == 1 and day == 1):
        event = "NewYear"

    return season, event


@app.route("/heatmap")
def heatmap():
    heat_data = []

    # get current day/hour
    now = datetime.now()
    current_day = now.strftime("%A")   # e.g. "Monday"
    current_hour = now.hour

    for _, row in stops_df.iterrows():
        try:
            # detect season + event dynamically
            season, event = get_current_season_and_event()
            print(f"📅 Using Season={season}, Event={event}")  # debug log

            X_pred = pd.DataFrame([[row["Stop"], current_day, current_hour, season, event]],
                                  columns=["Stop", "DayOfWeek", "Hour", "Season", "Event"])
            X_pred[["Stop", "DayOfWeek", "Season", "Event"]] = encoder.transform(
                X_pred[["Stop", "DayOfWeek", "Season", "Event"]]
            )

            vol = model.predict(X_pred)[0]
            heat_data.append([row["Latitude"], row["Longitude"], vol])

        except Exception as e:
            print(f"❌ Failed encoding: Stop={row['Stop']} → {e}")

    print(f"🔥 Heatmap points: {len(heat_data)} for {current_day} {current_hour}:00")

    m = folium.Map(location=[7.07, 125.61], zoom_start=12)
    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)

    return m._repr_html_()

if __name__ == "__main__":
    app.run(debug=True)
