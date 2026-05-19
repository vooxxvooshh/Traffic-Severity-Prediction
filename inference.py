import numpy as np
import pandas as pd

from feature_engineering import add_interaction_features, build_base_features
from preprocessing import encode_categoricals, impute_and_scale


def row_from_payload(payload: dict, artifacts: dict) -> pd.DataFrame:
    hour = int(payload.get("hour", 12))
    dayofweek = int(payload.get("dayofweek", 0))
    street = payload.get("street", "")

    row = {
        "Start_Time": f"2024-01-01 {hour:02d}:00:00",
        "Street": street,
        "State": payload.get("state", "CA"),
        "Temperature(F)": float(payload["temperature"]),
        "Humidity(%)": float(payload["humidity"]),
        "Visibility(mi)": float(payload["visibility"]),
        "Wind_Speed(mph)": float(payload["wind_speed"]),
        "Precipitation(in)": float(payload.get("precipitation", 0)),
        "Pressure(in)": float(payload.get("pressure", 29.9)),
        "Distance(mi)": float(payload.get("distance", 0.1)),
        "Weather_Condition": payload["weather_condition"],
        "Sunrise_Sunset": payload.get("sunrise_sunset", "Day"),
        "Amenity": int(payload.get("amenity", False)),
        "Bump": int(payload.get("bump", False)),
        "Crossing": int(payload.get("crossing", False)),
        "Junction": int(payload.get("junction", False)),
        "Traffic_Signal": int(payload.get("traffic_signal", False)),
    }
    df = pd.DataFrame([row])
    df["hour"] = hour
    df["dayofweek"] = dayofweek
    return df


def build_feature_vector(payload: dict, artifacts: dict) -> np.ndarray:
    df = row_from_payload(payload, artifacts)
    df = build_base_features(df)
    df, _ = encode_categoricals(df, artifacts)
    df = add_interaction_features(df)

    feature_cols = artifacts["feature_names_all"]
    X = df[feature_cols]
    X_scaled, _, _ = impute_and_scale(
        X,
        X,
        {"imputer": artifacts["imputer"], "scaler": artifacts["scaler"]},
    )
    mask = artifacts.get("feature_mask")
    if mask is not None:
        X_scaled = X_scaled[:, mask]
    return X_scaled
