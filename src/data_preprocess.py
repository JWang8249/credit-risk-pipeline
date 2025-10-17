import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_and_preprocess(path="data/raw/credit_data.csv"):
    """
    Load and preprocess the credit risk dataset.
    - Reads the dataset
    - Scales numerical features
    - Splits into train/test sets
    - Saves the processed version
    """
    df = pd.read_csv(path)
    df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

    X = df.drop(columns=['ID', 'default'])
    y = df['default']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    processed = pd.DataFrame(X_scaled, columns=X.columns)
    processed["target"] = y.values
    processed.to_csv("data/processed/credit_data_clean.csv", index=False)

    return X_train, X_test, y_train, y_test, scaler
