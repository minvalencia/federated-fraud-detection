FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/models /app/data && \
    chmod 777 /app/models  # Ensure write permissions for models directory

# Install pip dependencies with retries
COPY requirements.txt .

# Configure pip to use a longer timeout and retry failed downloads
RUN pip config set global.timeout 1000 && \
    pip config set global.retries 10

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/

# Set environment variables for paths
ENV MODEL_PATH=/app/models \
    UPLOAD_PATH=/app/uploads \
    PYTHONPATH=/app

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]