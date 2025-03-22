# Deployment Guide for Fraud Detection API

## Overview
This guide covers the deployment process for the Fraud Detection API service on Render, including both the FastAPI application and the machine learning components.

## Prerequisites
1. Render account
2. Git repository with your code
3. Python 3.8+
4. CUDA-compatible GPU (optional, for faster training)

## Project Structure
```
fraud-detection-api/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── ml_model.py
│   │   └── utils/
│   ├── client/
│   │   └── client.py
│   └── utils/
│       └── data_preprocessor.py
├── models/
│   ├── fraud_detector.pt
│   └── scaler.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

## Local Testing

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI application:
```bash
uvicorn src.api.main:app --reload
```

3. Test the API endpoints:
```bash
curl -X POST -H "X-API-Key: test-key" -F "file=@test_data.csv" http://localhost:8000/upload
```

## Deployment Steps

### 1. Prepare for Deployment

1. Update `requirements.txt`:
```
fastapi==0.68.1
uvicorn==0.15.0
python-multipart==0.0.5
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
torch==1.9.0
```

2. Configure environment variables in Render:
```
PYTHON_VERSION=3.8
API_KEY=your-secure-api-key
MODEL_PATH=/opt/render/project/src/models
CUDA_VISIBLE_DEVICES=0  # if GPU is available
```

3. Update Dockerfile:
```dockerfile
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.8 python3-pip

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/uploads /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Deploy on Render

1. Create a new Web Service
2. Connect your Git repository
3. Configure build settings:
   - Environment: Docker
   - Build Command: `docker build -t fraud-detection-api .`
   - Start Command: `docker run -p 8000:8000 fraud-detection-api`

### 3. ML Model Management

1. Model Storage:
   - Models are saved in the `/app/models` directory
   - Use versioning for model files (e.g., `fraud_detector_v1.pt`)
   - Implement model backup and recovery

2. Training Configuration:
   - Set appropriate batch sizes based on available memory
   - Configure early stopping parameters
   - Implement model checkpointing

3. Inference Optimization:
   - Use batch processing for large prediction requests
   - Implement caching for frequent predictions
   - Monitor GPU memory usage

## Performance Optimization

### 1. API Performance

1. Caching:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

2. Batch Processing:
```python
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile]):
    predictions = []
    for file in files:
        # Process files in parallel
        prediction = await process_file(file)
        predictions.append(prediction)
    return predictions
```

### 2. ML Performance

1. Model Optimization:
```python
# Convert model to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("models/optimized_model.pt")

# Use half-precision for faster inference
model.half()
```

2. Data Pipeline Optimization:
```python
# Use parallel processing for data preprocessing
from multiprocessing import Pool

def process_data_parallel(data_chunks):
    with Pool() as pool:
        results = pool.map(preprocess_chunk, data_chunks)
    return pd.concat(results)
```

## Monitoring and Logging

### 1. Application Monitoring

1. Health Checks:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": model_manager.health_check(),
        "disk_usage": get_disk_usage(),
        "memory_usage": get_memory_usage()
    }
```

2. Logging Configuration:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. ML Monitoring

1. Model Performance Metrics:
```python
from prometheus_client import Counter, Histogram

PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time for prediction')
FRAUD_PREDICTIONS = Counter('fraud_predictions_total', 'Total fraud predictions')
```

2. Drift Detection:
```python
def monitor_data_drift(new_data, reference_data):
    drift_metrics = calculate_drift_metrics(new_data, reference_data)
    if drift_metrics['drift_detected']:
        notify_drift_detection(drift_metrics)
```

## Security Configuration

1. API Key Authentication:
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401)

    return await call_next(request)
```

2. File Upload Security:
```python
def validate_file(file: UploadFile):
    # Check file size
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large")

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type")
```

## Troubleshooting

### Common Issues

1. Model Loading Errors:
```python
try:
    model = torch.load("models/fraud_detector.pt")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fall back to backup model
    model = torch.load("models/fraud_detector_backup.pt")
```

2. Memory Issues:
```python
def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    if memory_usage > 1000:  # 1GB threshold
        logger.warning("High memory usage detected")
```

### Recovery Procedures

1. Model Recovery:
```python
def recover_model():
    try:
        # Try loading latest backup
        model = load_backup_model()
        return model
    except:
        # Fall back to base model
        return initialize_base_model()
```

2. Data Recovery:
```python
def recover_data():
    try:
        # Attempt to recover from backup
        data = load_backup_data()
        return data
    except:
        logger.error("Failed to recover data")
        raise HTTPException(status_code=500)
```

## Maintenance

### Regular Tasks

1. Model Updates:
```bash
# Schedule regular model retraining
0 0 * * 0 python3 /app/src/scripts/retrain_model.py

# Backup models weekly
0 0 * * 1 python3 /app/src/scripts/backup_models.py
```

2. Performance Monitoring:
```bash
# Monitor system metrics hourly
0 * * * * python3 /app/src/scripts/monitor_performance.py

# Clean up old logs daily
0 0 * * * find /app/logs -type f -mtime +7 -delete
```

## Support and Documentation

For additional support:
- Technical documentation: https://docs.your-company.com
- API documentation: https://api.your-company.com/docs
- Support email: support@your-company.com
- Status page: https://status.your-company.com