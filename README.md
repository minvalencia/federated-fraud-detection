# Bank Transaction Fraud Detection API

A sophisticated machine learning-based API service for detecting fraudulent bank transactions using neural networks and behavioral analysis.

## 🌟 Features

### 1. Advanced Fraud Detection
- Neural network model with dynamic threshold adjustment
- Real-time and batch transaction processing
- Behavioral pattern analysis
- Feature importance explanation
- Federated learning support for multi-bank collaboration

### 2. Smart Data Processing
- Automated feature engineering
- Intelligent column mapping
- Robust data validation
- Support for multiple data formats
- Outlier detection and handling

### 3. API Capabilities
- RESTful endpoints with FastAPI
- Secure token-based authentication
- Real-time transaction scoring
- Batch processing support
- Comprehensive error handling
- Detailed performance metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Docker (optional)
- GPU support (optional, for faster training)

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup
```bash
# Build and start with Docker Compose
docker-compose up -d

# Check logs
docker logs fraud-api
```

## 📚 Project Structure
```
fraud-detection-api/
├── src/
│   ├── api/            # FastAPI application
│   ├── client/         # Federated learning client
│   ├── models/         # ML model implementations
│   ├── server/         # Federated learning server
│   ├── tests/          # Unit and integration tests
│   └── utils/          # Helper utilities
├── data/               # Data storage
├── models/             # Trained model storage
└── docs/               # Documentation files
```

## 🔧 API Endpoints

### Authentication
All endpoints (except /health) require API token:
```http
X-API-Key: your-api-token
```

### 1. Health Check
```http
GET /health
```

### 2. File Upload
```http
POST /upload
Content-Type: multipart/form-data
```

### 3. Data Processing
```http
POST /process/{filename}
Content-Type: application/json
```

### 4. Model Training
```http
POST /train/{filename}
```

### 5. Fraud Detection
```http
POST /predict
Content-Type: application/json

{
    "transaction_amount": 5000.0,
    "account_balance": 1000.0,
    "customer_age": 35,
    "login_attempts": 2,
    "transaction_duration": 120
}
```

## 📊 Model Architecture

### Neural Network Structure
```python
FraudDetector(
    Linear(5, 32)    # Input layer
    ReLU()
    Dropout(0.2)
    Linear(32, 16)   # Hidden layer 1
    ReLU()
    Dropout(0.2)
    Linear(16, 8)    # Hidden layer 2
    ReLU()
    Linear(8, 1)     # Output layer
    Sigmoid()        # Probability output
)
```

### Feature Engineering
- Transaction amount normalization
- Account balance analysis
- Age-based risk factors
- Login attempt patterns
- Transaction duration analysis

## 🔒 Security Features

1. **Authentication**
   - Token-based API security
   - Rate limiting
   - Request validation

2. **Data Protection**
   - Input sanitization
   - Secure parameter handling
   - No raw data storage

3. **Model Security**
   - Federated learning support
   - Model versioning
   - Access control

## 📈 Performance Metrics

The system provides comprehensive metrics:
- Precision and Recall
- F1 Score
- ROC Curve data
- Feature importance
- Prediction explanations

## 🛠️ Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_api.py
pytest tests/test_model.py
```

### Code Style
```bash
# Check code style
flake8 src/

# Format code
black src/
```

## 📝 Documentation

Detailed documentation is available in:
- [API Documentation](API_DOCUMENTATION.md)
- [Implementation Details](IMPLEMENTATION_DETAILS.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Postman Guide](POSTMAN_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support:
1. Check the documentation
2. Open an issue
3. Contact the development team

## 🔄 Updates and Maintenance

### Version History
- v1.0.0 - Initial release
- v1.1.0 - Added federated learning
- v1.2.0 - Dynamic threshold adjustment
- v1.3.0 - Enhanced feature engineering

### Roadmap
- [ ] Advanced anomaly detection
- [ ] Real-time model updates
- [ ] Enhanced API monitoring
- [ ] Additional ML models