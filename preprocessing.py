import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(path):

    use_cols = [
        'Severity',
        'Temperature(F)',
        'Humidity(%)',
        'Visibility(mi)',
        'Wind_Speed(mph)',
        'Weather_Condition',
        'Amenity',
        'Bump',
        'Crossing',
        'Junction',
        'Traffic_Signal'
    ]

    # ✅ ONLY LOAD REQUIRED COLUMNS + LIMIT SIZE
    df = pd.read_csv(
        path,
        usecols=use_cols,
        nrows=300000,   # 🔥 IMPORTANT FIX (prevents crash)
        low_memory=False
    )

    # Remove missing values
    df = df.dropna()

    # Optional but IMPORTANT: balanced sampling
    df = df.sample(n=100000, random_state=42)
    # Encode categorical column
    le = LabelEncoder()
    df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])

    # Split features and target
    X = df.drop('Severity', axis=1)
    y = df['Severity']

    # Convert to float for stability
    X = X.astype('float32')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns, le