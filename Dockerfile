# ---------- 1. Base image ----------
FROM python:3.10-slim

# ---------- 2. Set working directory ----------
WORKDIR /app

# ---------- 3. Copy project files ----------
COPY . /app

# ---------- 4. Install dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 5. Expose FastAPI & Streamlit ports ----------
EXPOSE 8000 8501

# ---------- 6. Default command ----------
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
