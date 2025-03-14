# Federated Fraud Detection System

A privacy-preserving federated learning system for detecting fraudulent banking transactions across multiple banks without sharing sensitive customer data.

## Overview

This system enables banks to collaboratively train a fraud detection model while keeping their transaction data private. It uses federated learning to share model improvements without exposing raw data.

### Key Features
- 🔒 Privacy-preserving federated learning
- 🎯 Advanced fraud detection scoring
- 📊 Real-time model updates
- 🏦 Multi-bank collaboration
- 📈 Dynamic threshold adaptation

## System Architecture

```
                    ┌─────────────────┐
                    │  Global Server  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────┴─────────┐         ┌────────┴──────────┐
    │    Client Bank A  │         │   Client Bank B   │
    └─────────┬─────────┘         └────────┬──────────┘
              │                             │
     [Private Data A]                [Private Data B]
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Flower (flwr)
- pandas
- scikit-learn
- numpy

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd federated-fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Start the Server
```bash
python src/server/server.py
```

### 2. Start Clients (in separate terminals)
```bash
# Client 0
python src/client/client.py --client_id 0 --num_clients 3 --data_path data/bank_transactions_data.csv

# Client 1
python src/client/client.py --client_id 1 --num_clients 3 --data_path data/bank_transactions_data.csv

# Client 2
python src/client/client.py --client_id 2 --num_clients 3 --data_path data/bank_transactions_data.csv
```

## Fraud Detection System

### Scoring Mechanism
The system uses a weighted scoring approach to identify potential fraud:

```python
fraud_score = (
    2.0 * (TransactionAmount > 90th_percentile) +  # High-value transaction
    1.5 * (LoginAttempts > 2) +                    # Multiple login attempts
    1.0 * (TransactionDuration > 90th_percentile) + # Unusual duration
    1.5 * (AccountBalance < 10th_percentile)        # Low balance
)
```

### Performance Metrics
Current system performance:

```
Client 0:
- Accuracy: 93.45%
- Precision: 26.67%
- Recall: 100%
- F1 Score: 42.11%

Client 2:
- Accuracy: 95.24%
- Precision: 45.45%
- Recall: 71.43%
- F1 Score: 55.56%
```

## Model Architecture

### Neural Network Structure
```python
FraudDetector(
    Input(5) → Dense(32) → ReLU → Dropout(0.2) →
    Dense(16) → ReLU → Dropout(0.2) →
    Dense(1) → Sigmoid
)
```

### Features Used
1. Transaction Amount (primary indicator)
2. Login Attempts (security indicator)
3. Account Balance (context indicator)
4. Transaction Duration (behavioral indicator)
5. Customer Age (risk factor)

## Privacy & Security

### Data Protection
- Raw transaction data never leaves the bank
- Only model parameters are shared
- Secure parameter transmission using TLS
- Independent data processing per client

### Federated Learning
- Local model training on bank's private data
- Aggregated model updates
- No direct access between clients
- Encrypted parameter exchange

## Configuration

### Server Settings
```python
{
    'host': '0.0.0.0',
    'port': 8081,
    'min_clients': 2,
    'rounds': 3
}
```

### Client Settings
```python
{
    'batch_size': 32,
    'max_epochs': 10,
    'learning_rate': 0.001,
    'early_stopping_patience': 2
}
```

## Project Structure
```
federated-fraud-detection/
├── src/
│   ├── server/
│   │   ├── server.py
│   │   └── __init__.py
│   ├── client/
│   │   ├── client.py
│   │   └── __init__.py
│   └── models/
│       ├── fraud_detector.py
│       └── __init__.py
├── data/
│   └── bank_transactions_data.csv
├── docs/
│   ├── FRAUD_DETECTION_LOGIC.md
│   └── FEDERATED_LEARNING_PROCESS.md
├── requirements.txt
└── README.md
```

## Documentation

Detailed documentation is available in the `docs` folder:
- [Fraud Detection Logic](docs/FRAUD_DETECTION_LOGIC.md)
- [Federated Learning Process](docs/FEDERATED_LEARNING_PROCESS.md)

## Performance Monitoring

### Metrics Tracked
- Training loss
- Model accuracy
- Precision and recall
- F1 score
- Client participation
- Convergence rate

### Logging
```python
{
    'timestamp': '2025-03-15 00:03:37,877',
    'client_id': 0,
    'round': 1,
    'metrics': {
        'accuracy': 0.9345,
        'precision': 0.2667,
        'recall': 1.0000
    }
}
```

## Future Enhancements

### Planned Features
1. Dynamic class weight adjustment
2. Advanced feature engineering
3. Adaptive learning rates
4. Differential privacy implementation
5. Enhanced monitoring and alerts

### Roadmap
- [ ] Implement adaptive aggregation
- [ ] Add dynamic client selection
- [ ] Enhance privacy measures
- [ ] Improve convergence speed
- [ ] Add more fraud indicators

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flower (flwr) team for the federated learning framework
- PyTorch team for the deep learning framework
- Contributors and reviewers

## Contact

Project Link: [https://github.com/yourusername/federated-fraud-detection](https://github.com/yourusername/federated-fraud-detection)