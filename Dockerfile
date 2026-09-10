# Dockerfile khusus untuk Hugging Face Spaces
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [PENTING] Pre-download model saat proses build
# Ini mencegah timeout 30 menit karena Hugging Face tidak perlu mendownload model 2.3 GB saat aplikasi di-start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy source code
COPY . .

# Hugging Face Spaces WAJIB menggunakan port 7860
ENV PORT=7860
EXPOSE 7860

# Jalankan FastAPI di port 7860
CMD uvicorn main:app --host 0.0.0.0 --port 7860
