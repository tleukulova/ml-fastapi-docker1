# ─────────────────────────────────────────────────────────────────
# Dockerfile — Titanic ML FastAPI Service
# Builds a container that serves the trained model via FastAPI
# on port 1912.
# ─────────────────────────────────────────────────────────────────

# Use official Python slim image to keep the image lightweight
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (leverages Docker layer cache)
COPY requirements.txt .

# Install Python dependencies (no cache to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the API port
EXPOSE 1912

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1912"]
