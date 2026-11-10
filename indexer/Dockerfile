# Pinned base image (Python 3.12-slim)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for PDF parsing and OCR extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy indexer pipeline code and seed fixtures
COPY pipeline/ ./pipeline/
COPY fixtures/ ./fixtures/

# Create and switch to non-root user
RUN groupadd -g 10001 indexergroup && \
    useradd -u 10001 -g indexergroup -s /bin/false -m indexeruser && \
    mkdir -p /corpus && \
    chown -R indexeruser:indexergroup /app /corpus

USER indexeruser

# Exec-form command running the indexer CLI
CMD ["python", "-m", "pipeline.run"]
