# Bank Transaction Fraud Detection API

A sophisticated API service for detecting fraudulent bank transactions using machine learning and behavioral analysis.

## Features

### 1. Intelligent Fraud Detection
- Neural network-based fraud detection model
- Behavioral pattern analysis
- Feature importance explanation
- Configurable detection thresholds
- Real-time and batch processing capabilities

### 2. Smart Data Processing
- Automatic column mapping for different data formats
- Derived login attempt scoring from behavioral patterns
- Data validation and standardization
- Support for multiple bank data formats
- Comprehensive data quality checks

### 3. API Features
- RESTful endpoints for all operations
- Secure token-based authentication
- Real-time transaction scoring
- Batch processing support
- Detailed performance metrics
- Explainable AI features

## Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running with Docker
```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api
```

### Getting the API Token
```bash
# Make a GET request to
curl http://localhost:8000/token
```

## API Endpoints

### 1. Data Upload and Processing
```http
# Upload transaction data
POST /upload/
Content-Type: multipart/form-data
X-API-Key: your-api-token

# Process uploaded file
POST /process/{filename}
Content-Type: application/json
X-API-Key: your-api-token
```

### 2. Fraud Detection
```http
# Single transaction prediction
POST /predict
Content-Type: application/json
X-API-Key: your-api-token

{
    "Transaction_Amount": 5000.0,
    "Account_Balance": 1000.0,
    "Age": 35,
    "Device_Type": "Mobile",
    "Transaction_Location": "New York",
    "Transaction_Time": "14:30:00",
    "Transaction_Device": "iPhone"
}

# Batch predictions
POST /predict/batch/{filename}
X-API-Key: your-api-token
```

### 3. Model Analysis
```http
# Get model metrics
GET /model/metrics/{filename}
X-API-Key: your-api-token

# Get model status
GET /model/status
X-API-Key: your-api-token
```

## Data Format

### Required Columns
The API supports flexible column mapping but requires these data points:
- Transaction Amount
- Account Balance
- Age
- Device Information
- Location Data
- Transaction Time
- Device Type

### Example CSV Format
```csv
Transaction_Amount,Account_Balance,Age,Device_Type,Transaction_Location,Transaction_Time,Transaction_Device
5000.0,1000.0,35,Mobile,New York,14:30:00,iPhone
```

## Fraud Detection Logic

### 1. Login Attempts Score
Derived from:
- Device type risk assessment
- Location consistency checks
- Time-based patterns
- Device switching behavior

### 2. Transaction Risk Factors
- Unusual transaction amounts
- Account balance patterns
- Age-related risk factors
- Suspicious login activity

### 3. Model Features
- Normalized transaction amount
- Normalized account balance
- Standardized age
- Derived login attempts score

## Performance Metrics

The API provides comprehensive performance metrics:
- Precision
- Recall
- Specificity
- F1 Score
- ROC Curve data
- Confusion matrix

## Security

### Authentication
- Token-based authentication required for all endpoints
- Tokens are automatically generated and stored securely
- Rate limiting implemented

### Data Protection
- No raw data storage
- Encrypted communication
- Secure parameter handling

## Error Handling

The API provides detailed error messages for:
- Invalid data formats
- Missing required fields
- Authentication failures
- Processing errors
- Model prediction issues

## Development

### Project Structure
```
project/
├── data/
│   ├── Bank_Transaction_Fraud_Detection.csv
│   └── column_mappings.json
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI application
│   │   └── ml_model.py       # ML model implementation
│   ├── models/
│   │   └── fraud_detector.py # Fraud detection model
│   ├── utils/
│   │   └── data_preprocessor.py  # Data preprocessing logic
│   └── __init__.py
├── API_DOCUMENTATION.md
├── DEPLOYMENT.md
└── docker-compose.yml
```

### Local Development
```bash
# Run the API locally
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run tests
pytest tests/

# Test with sample data
python src/utils/test_preprocessor.py
```

## Deployment

### Docker Deployment
```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker logs fraud-api
```

### Production Considerations
- Use proper SSL/TLS certificates
- Implement proper logging
- Set up monitoring
- Configure backup systems
- Implement CI/CD pipelines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the repository or contact the development team.