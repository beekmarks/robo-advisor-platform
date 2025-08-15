FROM python:3.11-slim

WORKDIR /app

# Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY app.py ./

EXPOSE 8084

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8084"]
