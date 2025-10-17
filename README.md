# Credit Risk Prediction Pipeline

A complete and reproducible **Machine Learning pipeline** for predicting credit default risk — including **data preprocessing**, **model training**, **explainability with SHAP**, **FastAPI backend**, **Streamlit frontend**, and **PostgreSQL persistence**, all containerized via **Docker** and **automated with CI/CD**.

---

## 👤 Author

**Jingyi Wang**  
Master’s Student in *Data Science and Society (Business Track)*  
Tilburg University, The Netherlands  
Version: **v2.0.0** (2025-10)  
Status: ✅ Fully reproducible and containerized

---

## 🧩 Overview

This project demonstrates a **modular and reproducible ML workflow**:

1. 📊 **Data preprocessing** (`src/data_preprocess.py`)  
2. 🤖 **Model training** (`src/model_train.py`)  
3. 📈 **Evaluation + Explainability** (SHAP visualization)  
4. ⚙️ **FastAPI REST service** for predictions  
5. 🖥️ **Streamlit web app** for interactive user input  
6. 🗄️ **PostgreSQL** for saving predictions  
7. 🤮 **Pytest** for automated testing  
8. 🐳 **Docker & Docker Compose** for environment reproducibility  
9. 🚦 **CI/CD pipeline** with GitHub Actions

---

## ⚙️ Project Structure

```
credit-risk-pipeline/
├── data/
│   ├── processed/credit_data_clean.csv
│   ├── raw/credit_data.csv
├── docs/
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
├── src/
│   ├── api.py
│   ├── app.py
│   ├── data_preprocess.py
│   ├── model_train.py
│   ├── model_eval.py
│   └── test_model.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1️⃣ Prepare Environment
```bash
conda create -n creditrisk python=3.10
conda activate creditrisk
pip install -r requirements.txt
```

### 2️⃣ Preprocess Data
```bash
python src/data_preprocess.py
```

### 3️⃣ Train Model
```bash
make train
```

### 4️⃣ Evaluate with SHAP
```bash
make eval
```
Generated visuals:
- `docs/shap_summary.png`
- `docs/shap_feature_importance.png`

### 5️⃣ Run API
```bash
make serve
```
Visit 🔗 http://127.0.0.1:8000/docs

### 6️⃣ Run Streamlit App
```bash
make app
```
Visit 🔗 http://127.0.0.1:8501

---

## 🐳 Run with Docker

### 1️⃣ Build Image
```bash
docker build -t creditrisk-api .
```

### 2️⃣ Run API
```bash
docker run -p 8000:8000 creditrisk-api
```

### 3️⃣ Test Endpoint
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"LIMIT_BAL\":20000, \"SEX\":1, \"EDUCATION\":2, \"MARRIAGE\":1, \"AGE\":35}"
```

---

## 🧱 docker-compose: Multi-Container Setup

Run the full stack (API + Streamlit + PostgreSQL):

```bash
docker-compose up -d
```

Services:
| Service | Port | Description |
|----------|------|-------------|
| `creditrisk-db` | 5432 | PostgreSQL database |
| `creditrisk-api` | 8000 | FastAPI model server |
| `creditrisk-app` | 8501 | Streamlit frontend |

Then open:
- http://127.0.0.1:8000/docs → FastAPI
- http://127.0.0.1:8501 → Streamlit App

---

## 🤓 SHAP Explainability

The pipeline integrates **SHAP (SHapley Additive exPlanations)** to interpret model predictions.

### Example Output:
| File | Description |
|------|--------------|
| `docs/shap_summary.png` | Feature importance summary |
| `docs/shap_feature_importance.png` | Bar plot of top drivers |

```python
import shap
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)
shap.summary_plot(shap_values, X, show=False)
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

File: `.github/workflows/ci.yml`

### CI Stages:
1. **Checkout repo**
2. **Set up Python environment**
3. **Install dependencies**
4. **Run unit tests (pytest)**
5. **Build Docker image**

Badge example:
[![CI](https://github.com/tcsai/credit-risk-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/tcsai/credit-risk-pipeline/actions/workflows/ci.yml)

This ensures every commit is:
- ✅ Lint-checked
- ✅ Tested
- ✅ Docker-buildable

---

## 🤪 Testing

Run automated tests:
```bash
make test
```

Tests include:
- Model loading  
- Prediction correctness  
- API endpoint validation  
- Database integration  

---

## 📊 Example Prediction

Input:
```json
{
  "LIMIT_BAL": 20000,
  "SEX": 1,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35
}
```

Output:
```json
{
  "prediction": 0,
  "risk": "Low Risk"
}
```

All predictions are logged in PostgreSQL table `predictions`.

---

## 📦 Makefile Commands

| Command | Description |
|----------|--------------|
| `make train` | Train the model |
| `make eval` | Evaluate and generate SHAP |
| `make serve` | Start FastAPI server |
| `make app` | Run Streamlit frontend |
| `make test` | Run all pytest scripts |
| `make all` | Run full pipeline (train → eval → test) |

---

## 🧠 Key Learnings

- Building reproducible ML pipelines requires clear separation between **training**, **serving**, and **testing**.  
- Docker + CI/CD ensures deterministic builds.  
- SHAP provides interpretable insights essential for **Law + AI** applications.  

---

## 📜 License

Licensed under the **MIT License**.  
Feel free to fork and adapt for your own data science portfolio.

---