# 💳 Credit Risk Prediction Pipeline

A fully reproducible machine learning pipeline for **credit card default risk prediction**, built with **Python**, **scikit-learn**, **FastAPI**, **Streamlit**, and **Docker**.  
This project automates the process from data preprocessing to model training, evaluation, API deployment, and web app integration — and includes full Docker support for easy deployment.

---

## 🧠 Overview

This project predicts whether a credit card client will default next month based on financial and demographic data.

**Pipeline includes:**
1. Data preprocessing and feature scaling  
2. Model training with logistic regression  
3. Evaluation with precision, recall, and F1-score  
4. REST API backend (FastAPI)  
5. Web frontend (Streamlit)  
6. Docker containerization for reproducibility and deployment  

---

## 🧩 Project Structure

```
credit-risk-pipeline/
│
├── data/
│   ├── raw/
│   │   └── credit_data.csv
│   └── processed/
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── data_preprocess.py
│   ├── model_train.py
│   ├── model_eval.py
│   ├── api.py
│   └── app.py
│
├── tests/
│   └── test_model.py
│
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Local)

### 1️⃣ Create and activate conda environment
```bash
conda create -n creditrisk python=3.10 -y
conda activate creditrisk
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model
```bash
make train
```

### 4️⃣ Evaluate the model
```bash
make eval
```

---

## 🚀 Run Locally

### 🔹 Start FastAPI backend
```bash
make serve
```
Access API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 🔹 Start Streamlit frontend
```bash
make app
```
Access App: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Deployment

### 🧱 1. Build Docker Image
```bash
docker build -t creditrisk-app .
```

### ▶️ 2. Run with FastAPI
```bash
docker run -d -p 8000:8000 creditrisk-app uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Access API at: [http://localhost:8000/docs](http://localhost:8000/docs)

### ▶️ 3. Run with Streamlit
```bash
docker run -d -p 8501:8501 creditrisk-app streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
```
Access App at: [http://localhost:8501](http://localhost:8501)

---

## 🧰 Dockerfile Example

```dockerfile
# ---------- Base image ----------
FROM python:3.10-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Copy project files ----------
COPY . /app

# ---------- Install dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Expose ports ----------
EXPOSE 8000 8501

# ---------- Default command ----------
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## ⚙️ docker-compose.yml Example

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: creditrisk-api
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"

  app:
    build: .
    container_name: creditrisk-app
    command: streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
```

### ▶️ Run both services
```bash
docker-compose up
```
Then visit:  
- FastAPI → [http://localhost:8000/docs](http://localhost:8000/docs)  
- Streamlit → [http://localhost:8501](http://localhost:8501)

---

## 🧾 Example Output

| Scenario | Output | Meaning |
|-----------|---------|---------|
| Low-risk client | ✅ `{"prediction": 0, "risk": "Low Risk"}` | Client likely to repay |
| High-risk client | 🚨 `{"prediction": 1, "risk": "High Risk"}` | Client likely to default |

---

## 🧠 Author

**Jingyi Wang (Justin)**  
📍 Tilburg University — MSc Data Science and Society  
✉️ J.Wang@tilburguniversity.edu  
🐙 GitHub: [@JWang8249](https://github.com/JWang8249)

---

## 🧩 Future Improvements

- [ ] Integrate model explainability (SHAP, LIME)  
- [ ] Add PostgreSQL database for persistent storage  
- [ ] CI/CD pipeline with GitHub Actions  
- [ ] Deploy on AWS/GCP via Docker containers  

---

## 📄 License

Licensed under the **MIT License**.  
Feel free to use and modify for educational or research purposes.

---

> _“From raw data to deployed intelligence — now fully containerized.”_ 🚀