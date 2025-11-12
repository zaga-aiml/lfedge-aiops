# =====================================
# Base: Lightweight Python Image
# =====================================
FROM python:3.11-slim

# Install essential system packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create Non-Root Working Environment

RUN mkdir -p /app /tmp/huggingface /mnt/hf_cache && chmod -R 777 /app /tmp/huggingface /mnt/hf_cache

WORKDIR /app


# Environment Variables

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV PYTHONPATH=/app


# Copy Dependencies & Install

COPY requirements.txt .

# Explicit LangChain version fix (to avoid vectorstore import issues)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir langchain==0.2.16 langchain-community==0.2.16 langchain-core==0.2.38


# Copy Application Source

COPY . .


# Expose and Run

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "debug", "--access-log"]
