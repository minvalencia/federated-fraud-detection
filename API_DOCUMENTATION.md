# Fraud Detection API Documentation

## Overview
This API provides endpoints for processing bank transaction data and detecting potential fraud using machine learning. The API supports file upload, data processing, model training, and fraud prediction.

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
    "timestamp": "2025-03-22T12:00:00.000Z"
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
        "TransactionAmount": "Transaction_Amount",
        "AccountBalance": "Account_Balance",
        "CustomerAge": "Age",
        "TransactionDuration": "Duration",
        "LoginAttempts": "Attempts"
    }
}
```

**Response:**
```json
{
    "status": "success",
    "processed_filename": "processed_bank_data_20250322_010056.csv",
    "rows_processed": 2512,
    "columns_standardized": ["TransactionAmount", "amount_normalized", ...],
    "fraud_statistics": {
        "total_transactions": 2512,
        "fraud_transactions": 75,
        "fraud_percentage": 2.98
    }
}
```

### 5. Train Model
```http
POST /train/{filename}
```
Train the fraud detection model on processed data.

**Response:**
```json
{
    "status": "success",
    "message": "Model trained successfully",
    "training_stats": {
        "losses": [0.6932, 0.3245, 0.2156, ...]
    },
    "details": {
        "features_used": ["amount_normalized", "duration_normalized", ...],
        "total_samples": 2512,
        "fraud_samples": 75,
        "fraud_percentage": 2.98
    }
}
```

### 6. Predict Fraud
```http
POST /predict/{filename}
```
Make fraud predictions on processed data.

**Request Body:**
```json
{
    "threshold": 0.3
}
```

**Response:**
```json
{
    "status": "success",
    "predictions_summary": {
        "total_transactions": 2512,
        "fraud_detected": 75,
        "legitimate_transactions": 2437,
        "fraud_percentage": 2.98,
        "probability_stats": {
            "mean": 0.15,
            "median": 0.12,
            "std": 0.18,
            "min": 0.01,
            "max": 0.95,
            "percentiles": {
                "25": 0.05,
                "75": 0.25,
                "90": 0.45,
                "95": 0.65,
                "99": 0.85
            }
        }
    },
    "fraud_transactions": {
        "total_count": 75,
        "details_shown": 75,
        "details": [...]
    }
}
```

### 7. List Files
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

### 8. Model Status
```http
GET /model/status
```
Get the current status of the fraud detection model.

**Response:**
```json
{
    "status": "success",
    "model_info": {
        "input_dimensions": 5,
        "device": "cpu",
        "model_file_exists": true,
        "scaler_file_exists": true
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
- Rate limit headers included in responses

## Support
- Documentation: https://docs.your-company.com
- Support email: support@your-company.com
- Status page: https://status.your-company.com