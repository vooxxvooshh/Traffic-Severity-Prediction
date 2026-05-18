from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_model.pkl"
ENCODER_PATH = BASE_DIR / "weather_encoder.pkl"

app = FastAPI(
    title="Traffic Severity Predictor",
    description="Predict US traffic accident severity from weather and road context.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

model = joblib.load(MODEL_PATH)
weather_encoder = joblib.load(ENCODER_PATH)

FEATURE_ORDER = [
    "temperature",
    "humidity",
    "visibility",
    "wind_speed",
    "weather_encoded",
    "amenity",
    "bump",
    "crossing",
    "junction",
    "traffic_signal",
]


class InputData(BaseModel):
    temperature: float = Field(..., description="Temperature in Fahrenheit")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    visibility: float = Field(..., ge=0, description="Visibility in miles")
    wind_speed: float = Field(..., ge=0, description="Wind speed in mph")
    weather_condition: str = Field(..., description="Weather condition label")
    amenity: bool = False
    bump: bool = False
    crossing: bool = False
    junction: bool = False
    traffic_signal: bool = False


def encode_weather(condition: str) -> int:
    classes = list(weather_encoder.classes_)
    if condition not in classes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown weather condition '{condition}'. Use a value from the training set.",
        )
    return int(weather_encoder.transform([condition])[0])


def build_features(data: InputData) -> np.ndarray:
    return np.array(
        [
            [
                data.temperature,
                data.humidity,
                data.visibility,
                data.wind_speed,
                encode_weather(data.weather_condition),
                float(data.amenity),
                float(data.bump),
                float(data.crossing),
                float(data.junction),
                float(data.traffic_signal),
            ]
        ],
        dtype=np.float32,
    )


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/weather-options")
async def weather_options():
    return {"conditions": list(weather_encoder.classes_)}


@app.post("/predict")
async def predict(data: InputData):
    features = build_features(data)
    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0]),
        "features_used": FEATURE_ORDER,
    }
