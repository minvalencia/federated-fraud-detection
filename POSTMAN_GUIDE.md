# Fraud Detection API - Postman Guide

## Prerequisites
1. Install Postman from [https://www.postman.com/downloads/](https://www.postman.com/downloads/)
2. Ensure the Fraud Detection API is running (default: `http://localhost:8000`)
3. Have your API token ready

## Setup in Postman

### 1. Create a New Collection
1. Open Postman
2. Click "Collections" in the sidebar
3. Click "+" to create a new collection
4. Name it "Fraud Detection API"

### 2. Set Collection Variables
1. Click on the collection
2. Go to "Variables" tab
3. Add the following variables:
   ```
   base_url: http://localhost:8000
   api_token: your-api-token
   ```

### 3. Set Collection Authorization
1. Go to "Authorization" tab
2. Type: "API Key"
3. Key: "X-API-Key"
4. Value: "{{api_token}}"
5. Add to: "Header"

## API Endpoints Guide

### 1. Get API Token
```http
GET {{base_url}}/token
```
- No authentication required
- Save the returned token as your `api_token` variable

### 2. Upload Transaction Data
```http
POST {{base_url}}/upload/
```
**Headers:**
- X-API-Key: {{api_token}}

**Body:**
- Form-data
- Key: "file"
- Value: Select your CSV file
- Type: file

**Example Response:**
```json
{
    "filename": "transactions_20240321.csv",
    "status": "success"
}
```

### 3. Process Uploaded File
```http
POST {{base_url}}/process/{filename}
```
**Headers:**
- X-API-Key: {{api_token}}
- Content-Type: application/json

**Body:**
```json
{
    "column_mapping": {
        "amount": "Transaction_Amount",
        "balance": "Account_Balance",
        "age": "Age",
        "device": "Device_Type",
        "location": "Transaction_Location",
        "time": "Transaction_Time",
        "device_details": "Transaction_Device"
    }
}
```

**Example Response:**
```json
{
    "status": "success",
    "processed_rows": 1000,
    "standardized_columns": [
        "amount_normalized",
        "balance_normalized",
        "age_normalized",
        "device_encoded",
        "location_encoded"
    ]
}
```

### 4. Make Fraud Predictions
```http
POST {{base_url}}/predict/{filename}
```
**Headers:**
- X-API-Key: {{api_token}}
- Content-Type: application/json

**Query Parameters:**
- threshold: 0.5 (optional)

**Example Response:**
```json
{
    "status": "success",
    "output_filename": "predictions_transactions_20240321.csv",
    "predictions_summary": {
        "total_transactions": 1000,
        "fraud_detected": 50,
        "fraud_percentage": 5.0,
        "probability_stats": {
            "mean": 0.15,
            "median": 0.12,
            "std": 0.2
        }
    },
    "fraud_transactions": {
        "count": 50,
        "details": [
            {
                "index": 42,
                "probability": 0.95,
                "normalized_features": {
                    "amount_normalized": 2.5,
                    "balance_normalized": -1.2,
                    "age_normalized": 0.8,
                    "attempts_normalized": 2.1
                },
                "transaction_data": {
                    "Transaction_Amount": 5000.00,
                    "Account_Balance": 1000.00,
                    "Age": 35,
                    "Device_Type": "Mobile"
                }
            }
        ]
    }
}
```

### 5. Get Model Metrics
```http
GET {{base_url}}/evaluate/{filename}
```
**Headers:**
- X-API-Key: {{api_token}}

**Example Response:**
```json
{
    "status": "success",
    "metrics": {
        "accuracy": 0.93,
        "precision": 0.93,
        "recall": 0.87,
        "f1_score": 0.89,
        "true_positives": 435,
        "false_positives": 32,
        "true_negatives": 4892,
        "false_negatives": 65
    }
}
```

## Testing Steps

1. **Initial Setup**
   - Create the collection
   - Set variables
   - Set authorization

2. **Get API Token**
   - Send GET request to /token
   - Copy token to collection variables

3. **Upload Data**
   - Select a CSV file
   - Send POST request to /upload/
   - Save returned filename

4. **Process Data**
   - Use filename from previous step
   - Send POST request to /process/{filename}
   - Verify processing success

5. **Make Predictions**
   - Use processed filename
   - Send POST request to /predict/{filename}
   - Review fraud predictions

6. **Evaluate Results**
   - Send GET request to /evaluate/{filename}
   - Review model performance metrics

## Error Handling

Common HTTP Status Codes:
- 200: Success
- 400: Bad Request (invalid data/parameters)
- 401: Unauthorized (invalid/missing token)
- 404: Not Found (file not found)
- 422: Validation Error (invalid input format)
- 500: Server Error

Example Error Response:
```json
{
    "detail": "Error message describing the problem"
}
```

## Best Practices

1. **File Naming**
   - Use descriptive filenames
   - Include date/time in filename
   - Avoid special characters

2. **Data Validation**
   - Verify CSV format
   - Check required columns
   - Validate data types

3. **Error Handling**
   - Check response status
   - Log error messages
   - Implement retry logic

4. **Performance**
   - Use batch processing for large files
   - Monitor response times
   - Implement rate limiting

## Troubleshooting

1. **Authentication Issues**
   - Verify token is correct
   - Check token expiration
   - Ensure proper header format

2. **File Upload Issues**
   - Check file size limits
   - Verify file format
   - Ensure proper form-data

3. **Processing Errors**
   - Verify column mapping
   - Check data format
   - Review error messages

4. **Prediction Issues**
   - Check threshold values
   - Verify feature names
   - Review input data quality