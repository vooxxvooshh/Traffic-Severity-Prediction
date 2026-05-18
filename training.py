# training.py

from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, path="traffic_model.pkl"):
    joblib.dump(model, path)


def save_weather_encoder(encoder, path="weather_encoder.pkl"):
    joblib.dump(encoder, path)