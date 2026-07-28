# ─────────────────────────────────────────────
# Stage 1: builder — install dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for faiss-cpu (needs libgomp) + bs4
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix for clean copy
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: runtime — lean final image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY evaluate.py .
COPY pyproject.toml .

# Writable volumes for persistent data, index, and cross-encoder weights
VOLUME ["/app/data", "/app/index"]

# Non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Expose FastAPI port
EXPOSE 8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the API server
CMD ["python", "-m", "uvicorn", "src.adapters.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
