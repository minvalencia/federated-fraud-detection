# Postman Guide for Fraud Detection API

## Setup

1. **Import Collection**
   - Open Postman
   - Click "Import" button
   - Import the provided collection file

2. **Environment Setup**
   - Create a new environment
   - Add the following variables:
     ```
     BASE_URL: http://localhost:8000
     API_KEY: your_api_key_here
     ```

## Authentication

All requests (except health check) require the API key in the header:
```
X-API-Key: {{API_KEY}}
```

## Request Examples

### 1. Health Check
```http
GET {{BASE_URL}}/health
```
No authentication required.

### 2. Upload File
```http
POST {{BASE_URL}}/upload
```
**Headers:**
```
X-API-Key: {{API_KEY}}
```
**Body:**
- Form-data
- Key: `file`
- Value: Select CSV file

### 3. Validate Mapping
```http
GET {{BASE_URL}}/validate/bank_data_20250322_010056.csv
```
**Headers:**
```
X-API-Key: {{API_KEY}}
```

### 4. Process File
```http
POST {{BASE_URL}}/process/bank_data_20250322_010056.csv
```
**Headers:**
```
X-API-Key: {{API_KEY}}
Content-Type: application/json
```
**Body:**
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

### 5. Train Model
```http
POST {{BASE_URL}}/train/processed_bank_data_20250322_010056.csv
```
**Headers:**
```
X-API-Key: {{API_KEY}}
```

### 6. Predict Fraud
```http
POST {{BASE_URL}}/predict/processed_bank_data_20250322_010056.csv
```
**Headers:**
```
X-API-Key: {{API_KEY}}
Content-Type: application/json
```
**Body:**
```json
{
    "threshold": 0.3
}
```

### 7. List Files
```http
GET {{BASE_URL}}/files/
```
**Headers:**
```
X-API-Key: {{API_KEY}}
```

### 8. Model Status
```http
GET {{BASE_URL}}/model/status
```
**Headers:**
```
X-API-Key: {{API_KEY}}
```

## Testing Workflow

1. **Initial Setup**
   - Verify health check endpoint is working
   - Ensure API key is correctly set in environment

2. **Data Upload and Processing**
   ```
   Upload File → Validate Mapping → Process File
   ```
   - Upload your transaction data
   - Validate the column mapping
   - Process the file with correct mapping

3. **Model Training and Prediction**
   ```
   Train Model → Model Status → Predict Fraud
   ```
   - Train the model on processed data
   - Verify model status
   - Make predictions with desired threshold

4. **File Management**
   ```
   List Files → Check Processed Files
   ```
   - Monitor uploaded and processed files
   - Clean up old files if needed

## Response Examples

### Successful Upload Response
```json
{
    "filename": "bank_data_20250322_010056.csv",
    "columns": ["Transaction_Amount", "Account_Balance", "Age", ...],
    "rows": 2512,
    "message": "File uploaded successfully"
}
```

### Successful Processing Response
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

### Successful Prediction Response
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
    }
}
```

## Error Handling

Common error responses and their meanings:

1. **401 Unauthorized**
   ```json
   {
       "detail": "Invalid or missing API token"
   }
   ```
   - Check API key is correct
   - Verify header is properly set

2. **400 Bad Request**
   ```json
   {
       "detail": "Error processing CSV file: Missing required column"
   }
   ```
   - Verify input data format
   - Check column mapping

3. **404 Not Found**
   ```json
   {
       "detail": "File not found"
   }
   ```
   - Verify file exists
   - Check filename spelling

## Best Practices

1. **Environment Management**
   - Use separate environments for development/production
   - Never hardcode API keys
   - Use variables for repeated values

2. **Testing**
   - Test with various data sizes
   - Verify error handling
   - Check response formats
   - Monitor processing times

3. **Data Preparation**
   - Use valid CSV format
   - Include required columns
   - Clean data before upload
   - Keep reasonable file sizes

4. **Security**
   - Keep API key secure
   - Use HTTPS in production
   - Monitor access logs
   - Regular security audits