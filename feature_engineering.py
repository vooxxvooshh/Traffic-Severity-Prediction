import numpy as np
import pandas as pd


def _cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["Start_Time"], errors="coerce")
    out["hour"] = dt.dt.hour.fillna(12).astype(int)
    out["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour"], 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(out["dayofweek"], 7)
    return out


def add_road_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    street = out["Street"].fillna("").astype(str).str.upper()
    out["is_highway"] = street.str.contains(
        r"I-|US-|HWY|FREEWAY|INTERSTATE|TPKE|TURNPIKE", regex=True
    ).astype(int)
    out["is_ramp"] = street.str.contains(r"RAMP|EXIT|ON-RAMP|OFF-RAMP", regex=True).astype(
        int
    )
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_weather"] = out["hour_sin"] * out["Weather_Condition"]
    out["highway_junction"] = out["is_highway"] * out["Junction"]
    out["visibility_precip"] = out["Visibility(mi)"] * out["Precipitation(in)"]
    out["wind_humidity"] = out["Wind_Speed(mph)"] * (out["Humidity(%)"] / 100.0)
    out["distance_signal"] = out["Distance(mi)"] * out["Traffic_Signal"]
    return out


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    return add_road_features(df)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Use after categorical encoding so Weather_Condition is numeric."""
    df = build_base_features(df)
    return add_interaction_features(df)
