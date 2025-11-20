FROM python:3.11.1-slim

WORKDIR /app

COPY requirements.txt .
COPY fastapi_CV.py .
COPY best_model.pth .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fastapi_CV:app", "--host", "0.0.0.0", "--port", "8000"]
