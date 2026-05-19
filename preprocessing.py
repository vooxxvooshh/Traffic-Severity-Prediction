from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from feature_engineering import add_interaction_features, build_base_features

RAW_COLUMNS = [
    "Severity",
    "Start_Time",
    "Street",
    "State",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Pressure(in)",
    "Distance(mi)",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Amenity",
    "Bump",
    "Crossing",
    "Junction",
    "Traffic_Signal",
]

BOOL_COLUMNS = ["Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"]
NUMERIC_COLUMNS = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Pressure(in)",
    "Distance(mi)",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_highway",
    "is_ramp",
    "hour_weather",
    "highway_junction",
    "visibility_precip",
    "wind_humidity",
    "distance_signal",
]
TOP_STATES = 12


def resolve_data_path(path: str | None = None) -> str:
    candidates = [path, "US_Accidents_March23.csv", "data/US_Accidents_March23.csv"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find US_Accidents_March23.csv")


def load_raw_data(path: str, nrows: int = 300_000, sample_size: int = 120_000) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_COLUMNS, nrows=nrows, low_memory=False)
    df = df.dropna(subset=["Severity", "Weather_Condition", "Start_Time"])
    for col in BOOL_COLUMNS:
        df[col] = df[col].astype(int)

    if len(df) > sample_size:
        df = df.groupby("Severity", group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), max(500, int(sample_size * len(g) / len(df)))),
                random_state=42,
            )
        )
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    fitted = encoders or {}

    if "weather" not in fitted:
        fitted["weather"] = LabelEncoder()
        out["Weather_Condition"] = fitted["weather"].fit_transform(
            out["Weather_Condition"].astype(str)
        )
    else:
        le = fitted["weather"]
        known = set(le.classes_)
        out["Weather_Condition"] = out["Weather_Condition"].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    if "sunrise" not in fitted:
        fitted["sunrise"] = LabelEncoder()
        out["Sunrise_Sunset"] = fitted["sunrise"].fit_transform(
            out["Sunrise_Sunset"].astype(str)
        )
    else:
        le = fitted["sunrise"]
        known = set(le.classes_)
        out["Sunrise_Sunset"] = out["Sunrise_Sunset"].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    top_states = fitted.get("top_states")
    if top_states is None:
        top_states = out["State"].value_counts().head(TOP_STATES).index.tolist()
        fitted["top_states"] = top_states
    out["State"] = out["State"].where(out["State"].isin(top_states), "OTHER")
    if "state" not in fitted:
        fitted["state"] = LabelEncoder()
        out["State"] = fitted["state"].fit_transform(out["State"].astype(str))
    else:
        le = fitted["state"]
        known = set(le.classes_)
        out["State"] = out["State"].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    return out, fitted


def prepare_features(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = build_base_features(df)
    df, encoders = encode_categoricals(df, encoders)
    df = add_interaction_features(df)

    feature_cols = (
        NUMERIC_COLUMNS
        + ["Weather_Condition", "Sunrise_Sunset", "State"]
        + BOOL_COLUMNS
    )
    X = df[feature_cols].copy()
    y = df["Severity"].astype(int)
    return X, y, encoders


def impute_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    artifacts: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fitted = artifacts or {}
    imputer = fitted.get("imputer") or SimpleImputer(strategy="median")
    scaler = fitted.get("scaler") or StandardScaler()

    X_train_imp = imputer.fit_transform(X_train) if "imputer" not in fitted else imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    fitted["imputer"] = imputer

    X_train_scaled = scaler.fit_transform(X_train_imp) if "scaler" not in fitted else scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    fitted["scaler"] = scaler

    return X_train_scaled, X_test_scaled, fitted


def select_features_by_importance(
    X_train: np.ndarray,
    y_train: pd.Series,
    feature_names: list[str],
    threshold: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from sklearn.ensemble import RandomForestClassifier

    selector = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    keep_mask = importances >= threshold
    if keep_mask.sum() < 8:
        top_idx = np.argsort(importances)[-12:]
        keep_mask = np.zeros_like(importances, dtype=bool)
        keep_mask[top_idx] = True

    selected = [name for name, keep in zip(feature_names, keep_mask) if keep]
    return X_train[:, keep_mask], keep_mask, selected


def load_and_preprocess_data(
    path: str | None = None,
    apply_feature_selection: bool = True,
) -> dict:
    data_path = resolve_data_path(path)
    df = load_raw_data(data_path)
    X, y, encoders = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = list(X.columns)
    X_train_scaled, X_test_scaled, scale_artifacts = impute_and_scale(X_train, X_test)
    encoders.update(scale_artifacts)
    encoders["feature_names_all"] = feature_names

    selected_names = feature_names
    if apply_feature_selection:
        X_train_scaled, keep_mask, selected_names = select_features_by_importance(
            X_train_scaled, y_train, feature_names
        )
        X_test_scaled = X_test_scaled[:, keep_mask]
        encoders["feature_mask"] = keep_mask

    encoders["feature_names"] = selected_names

    class_counts = y.value_counts().to_dict()
    encoders["class_distribution"] = class_counts

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "artifacts": encoders,
    }
