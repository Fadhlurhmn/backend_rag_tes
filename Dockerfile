# Dockerfile untuk Hugging Face Spaces
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (termasuk folder data)
COPY . .

# Hugging Face Spaces mengharuskan aplikasi berjalan di port 7860
EXPOSE 7860

# Jalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
