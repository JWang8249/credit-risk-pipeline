from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd

def evaluate_model(test_path="data/processed/credit_data_clean.csv"):
    """
    Load the trained model and evaluate on the processed dataset.
    """
    df = pd.read_csv(test_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model()
