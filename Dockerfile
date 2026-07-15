# Dockerfile for deploying the Resume Screening backend to Hugging Face Spaces
# (Docker SDK). No changes needed to your existing app code -- this just
# packages it up with everything it needs to run.

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by some Python packages (e.g. for PDF handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (better Docker layer caching -- faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
