from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from data_preprocess import load_and_preprocess

def train_model():
    """
    Train a logistic regression model and save the artifacts.
    """
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model performance report:")
    print(classification_report(y_test, preds))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
