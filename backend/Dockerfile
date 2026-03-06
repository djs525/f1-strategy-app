FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps — single RUN layer to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Python deps first — cached layer unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# 2 workers = 2 model copies in RAM (~1.5 GB each on this model).
# Safe for a 4 GB Droplet. Drop to 1 worker for a 2 GB Droplet.
CMD ["uvicorn", "phase_2.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]