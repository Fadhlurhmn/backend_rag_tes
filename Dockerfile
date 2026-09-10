# Dockerfile untuk Hugging Face Spaces
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (termasuk folder data)
COPY . .

# Railway akan memberikan port secara dinamis lewat environment variable $PORT
# (Default fallback ke 8000 jika $PORT tidak ada)
ENV PORT=8000

# Jalankan FastAPI (gunakan format shell agar bisa membaca $PORT)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
