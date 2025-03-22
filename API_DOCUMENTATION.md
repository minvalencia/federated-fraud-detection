# Fraud Detection API Documentation

## Overview
This API provides endpoints for processing bank transaction data and detecting potential fraud using machine learning. The API supports file upload, data processing, model training, and fraud prediction.

## Base URL
```
http://localhost:8000
```

## Authentication
All endpoints (except `/health`) require API token authentication.

**Header Required:**
- Key: `X-API-Key`
- Value: Your API token

## Endpoints

### 1. Health Check
```http
GET /health
```
Check if the API is running.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### 2. Upload File
```http
POST /upload
```
Upload a CSV file containing transaction data.

**Request:**
- Form-data with key "file" and a CSV file value

**Response:**
```json
{
    "filename": "bank_data_20250322_010056.csv",
    "columns": ["Transaction_Amount", "Account_Balance", "Age", ...],
    "rows": 2512,
    "message": "File uploaded successfully"
}
```

### 3. Validate Mapping
```http
GET /validate/{filename}
```
Validate the uploaded file and automatically map columns.

**Response:**
```json
{
    "mapped_columns": {
        "TransactionAmount": "Transaction_Amount",
        "AccountBalance": "Account_Balance",
        ...
    },
    "numeric_validation": {
        "invalid_columns": [],
        "valid_columns": ["TransactionAmount", "AccountBalance", ...]
    },
    "data_stats": {
        "TransactionAmount": {
            "mean": 1000.0,
            "min": 10.0,
            "max": 5000.0,
            "null_count": 0
        },
        ...
    }
}
```

### 4. Process File
```http
POST /process/{filename}
```
Process the file and prepare it for fraud detection.

**Request Body:**
```json
{
    "mapping": {
        "TransactionAmount": "amount",
        "TransactionDuration": "duration",
        "LoginAttempts": "attempts",
        "AccountBalance": "balance",
        "CustomerAge": "age"
    }
}
```

**Response:**
```json
{
    "status": "success",
    "processed_filename": "processed_bank_data_20250322.csv",
    "rows_processed": 2512,
    "columns_standardized": [
        "amount_normalized",
        "duration_normalized",
        "attempts_normalized",
        "balance_normalized",
        "age_normalized"
    ]
}
```

### 5. Train Model
```http
POST /train/{filename}
```
Train the fraud detection model. This endpoint will:
1. Create and save a StandardScaler for feature normalization
2. Train the neural network model
3. Save both the model and scaler for future predictions

**Response:**
```json
{
    "status": "success",
    "message": "Training started",
    "monitor_url": "/train/status"
}
```

### 6. Training Status
```http
GET /train/status
```
Get the current status of model training.

**Response:**
```json
{
    "status": "success",
    "training_status": {
        "is_training": false,
        "progress": 100,
        "current_epoch": 50,
        "total_epochs": 50,
        "current_loss": 0.0123,
        "best_loss": 0.0120,
        "message": "Training completed successfully",
        "start_time": "2025-03-22T13:57:16",
        "end_time": "2025-03-22T13:58:16"
    }
}
```

### 7. Predict Fraud
```http
POST /predict/{filename}?threshold=0.3
```
Make fraud predictions on processed data. Requires both model and scaler to be properly trained and saved.

**Query Parameters:**
- threshold (optional): Classification threshold (default: 0.5)

**Response:**
```json
{
    "status": "success",
    "predictions": {
        "fraud_count": 25,
        "total_transactions": 1000,
        "fraud_rate": 0.025,
        "threshold_used": 0.3,
        "results": [0, 1, 0, ...]
    }
}
```

### 8. List Files
```http
GET /files/
```
List all uploaded and processed files.

**Response:**
```json
{
    "files": [
        {
            "filename": "bank_data_20250322_010056.csv",
            "size_bytes": 1234567,
            "uploaded_at": "2025-03-22T01:00:56",
            "is_processed": false
        },
        {
            "filename": "processed_bank_data_20250322_010056.csv",
            "size_bytes": 2345678,
            "uploaded_at": "2025-03-22T01:01:30",
            "is_processed": true
        }
    ]
}
```

### 9. Model Status
```http
GET /model/status
```
Get the current status of the model and its components.

**Response:**
```json
{
    "status": "success",
    "model_info": {
        "model_file_exists": true,
        "scaler_file_exists": true,
        "input_dimensions": 5,
        "device": "cpu",
        "last_training": "2025-03-22T13:58:16"
    }
}
```

## Error Handling
The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (invalid input)
- 401: Unauthorized (invalid/missing API token)
- 404: Not Found
- 500: Internal Server Error

Error responses include a detail message explaining the error:
```json
{
    "detail": "Error message here"
}
```

## Data Requirements

### Required Columns
1. TransactionAmount
2. TransactionDuration
3. LoginAttempts
4. AccountBalance
5. CustomerAge

### Column Mapping
The API supports various column name formats:
```python
{
    'TransactionAmount': ['amount', 'transaction_value', 'amt'],
    'TransactionDuration': ['duration', 'processing_time'],
    'LoginAttempts': ['attempts', 'login_tries'],
    'AccountBalance': ['balance', 'current_balance'],
    'CustomerAge': ['age', 'customer_age']
}
```

## Best Practices
1. Always validate column mapping before processing
2. Monitor fraud detection metrics regularly
3. Adjust threshold based on your risk tolerance
4. Keep processed files for audit purposes
5. Implement proper error handling

## Rate Limiting
- 100 requests per minute per API key
- Training endpoints limited to 1 request per minute

## Support
- Documentation: https://docs.your-company.com
- Support email: support@your-company.com
- Status page: https://status.your-company.com