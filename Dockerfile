# ── Stage 1: build / install dependencies ─────────────────────────────────────
FROM python:3.11-slim AS builder

# System libs needed by OpenCV and rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into a local prefix so Stage 2 can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Same system libs (runtime only, no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy module source
COPY damage_check/ ./damage_check/

# Output directory (can be overridden with a volume mount)
RUN mkdir -p /app/output

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Default: start the API service with uvicorn
CMD ["uvicorn", "damage_check.service.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]

