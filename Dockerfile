FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ai_core.py .
COPY backend_api.py .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "7860"]