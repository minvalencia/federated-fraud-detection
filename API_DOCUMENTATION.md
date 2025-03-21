# Fraud Detection API Documentation

## Overview
This API service provides comprehensive fraud detection capabilities for bank transaction data, including data preprocessing, model training, and fraud prediction. It features intelligent column mapping, data validation, and detailed fraud analysis.

## Base URL
```
https://your-app-name.onrender.com
```

## Authentication
All endpoints require API key authentication:
```
X-API-Key: your-api-key-here
```

## Endpoints

### Data Upload and Processing

#### Upload File
- **POST** `/upload/`
- Uploads and analyzes a bank transaction file
- Request: `multipart/form-data` with file
- Response:
```json
{
    "status": "success",
    "filename": "bank_data_20250315_123456.csv",
    "total_rows": 1000,
    "column_mapping": {
        "TransactionAmount": "amount",
        "TransactionDuration": "duration",
        "LoginAttempts": "attempts",
        "AccountBalance": "balance",
        "CustomerAge": "age"
    },
    "mapping_coverage": 0.8,
    "invalid_columns": [],
    "missing_required_columns": []
}
```

#### Validate Mapping
- **GET** `/validate/{filename}`
- Validates column mapping and provides data statistics
- Parameters:
  - `mapping`: Dictionary of column mappings
- Response:
```json
{
    "status": "success",
    "invalid_columns": [],
    "statistics": {
        "TransactionAmount": {
            "mean": 1000.50,
            "median": 750.25,
            "min": 10.00,
            "max": 5000.00,
            "null_count": 0
        }
    }
}
```

#### Process File
- **POST** `/process/{filename}`
- Processes file using provided column mapping
- Request:
```json
{
    "mapping": {
        "TransactionAmount": "amount",
        "TransactionDuration": "duration"
    }
}
```
- Response:
```json
{
    "status": "success",
    "processed_filename": "processed_bank_data.csv",
    "rows_processed": 1000,
    "columns_standardized": ["TransactionAmount", "TransactionDuration"]
}
```

### Machine Learning Operations

#### Train Model
- **POST** `/train/{filename}`
- Trains the fraud detection model on processed data
- Response:
```json
{
    "status": "success",
    "message": "Model trained successfully",
    "training_stats": {
        "losses": [0.6932, 0.5421, 0.4120, 0.3856]
    }
}
```

#### Predict Fraud
- **POST** `/predict/{filename}`
- Parameters:
  - `threshold` (float, optional): Probability threshold (default: 0.5)
- Makes fraud predictions with detailed analysis
- Response:
```json
{
    "status": "success",
    "output_filename": "predictions_bank_data.csv",
    "predictions_summary": {
        "total_transactions": 1000,
        "fraud_detected": 50,
        "legitimate_transactions": 950,
        "fraud_percentage": 5.0,
        "probability_stats": {
            "mean": 0.15,
            "median": 0.08,
            "std": 0.25,
            "min": 0.01,
            "max": 0.98
        },
        "confidence_stats": {
            "avg_fraud_confidence": 0.85,
            "avg_legitimate_confidence": 0.92
        },
        "threshold_used": 0.5
    },
    "validation_status": "All validations passed",
    "fraud_transactions": {
        "count": 50,
        "details": [
            {
                "index": 123,
                "probability": 0.98,
                "normalized_features": {
                    "amount_normalized": 2.5,
                    "duration_normalized": 1.8,
                    "attempts_normalized": 3.2,
                    "balance_normalized": -0.5,
                    "age_normalized": 0.2
                },
                "transaction_data": {
                    "TransactionAmount": 5000.00,
                    "TransactionDuration": 120,
                    "LoginAttempts": 5,
                    "AccountBalance": 1000.00,
                    "CustomerAge": 35,
                    "TransactionType": "wire_transfer"
                }
            }
        ]
    }
}
```

#### Evaluate Model
- **GET** `/evaluate/{filename}`
- Evaluates model performance on processed data
- Response:
```json
{
    "status": "success",
    "metrics": {
        "accuracy": 0.95,
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.81,
        "true_positives": 45,
        "false_positives": 5,
        "true_negatives": 940,
        "false_negatives": 10
    }
}
```

#### Model Status
- **GET** `/model/status`
- Gets current model status and configuration
- Response:
```json
{
    "status": "success",
    "model_info": {
        "input_dimensions": 5,
        "device": "cuda",
        "model_file_exists": true,
        "scaler_file_exists": true
    }
}
```

### File Management

#### List Files
- **GET** `/files/`
- Lists all uploaded and processed files
- Response:
```json
{
    "files": [
        {
            "filename": "bank_data_20250315.csv",
            "size_bytes": 1024000,
            "uploaded_at": "2025-03-15T12:34:56",
            "is_processed": false
        }
    ]
}
```

## Error Handling
All endpoints use consistent error responses:
```json
{
    "detail": "Error message description"
}
```

Common status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

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