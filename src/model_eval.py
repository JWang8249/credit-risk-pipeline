from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

def evaluate_model(test_path="data/processed/credit_data_clean.csv"):
    """
    Load the trained model, evaluate on the processed dataset, 
    and visualize feature importance using SHAP.
    """
    # -------------------------------
    # 1. Load data and model
    # -------------------------------
    df = pd.read_csv(test_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # -------------------------------
    # 2. Evaluate model performance
    # -------------------------------
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification Report:")
    print(classification_report(y, preds))

    # -------------------------------
    # 3. SHAP Explainability
    # -------------------------------
    print("\nGenerating SHAP summary plot...")

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Create docs directory if not exist
    os.makedirs("docs", exist_ok=True)

    # Summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("docs/shap_summary.png", dpi=300)
    plt.close()

    # Bar plot of feature importance
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("docs/shap_feature_importance.png", dpi=300)
    plt.close()

    print("✅ SHAP summary and feature importance plots saved in /docs")

if __name__ == "__main__":
    evaluate_model()

