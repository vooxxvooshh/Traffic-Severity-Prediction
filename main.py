# main.py

from preprocessing import load_and_preprocess_data
from training import train_model, save_model, save_weather_encoder
from evaluation import evaluate_model

DATA_PATH = "data/US_Accidents_March23.csv"

def main():

    print("\n--- LOADING & PREPROCESSING DATA ---")
    X_train, X_test, y_train, y_test, features, weather_encoder = load_and_preprocess_data(DATA_PATH)

    print("\n--- TRAINING MODEL ---")
    model = train_model(X_train, y_train)

    print("\n--- SAVING MODEL ---")
    save_model(model)
    save_weather_encoder(weather_encoder)

    print("\n--- EVALUATING MODEL ---")
    y_pred = evaluate_model(model, X_test, y_test)

    print("\nPROJECT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()