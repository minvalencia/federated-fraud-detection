# Implementation Details

/*
Key Terms Explained:
- Neural Network: A machine learning model inspired by biological neural networks, used here for fraud detection.
- Dense Layer: A fully connected layer where each neuron is connected to all neurons in the previous layer.
- ReLU: Rectified Linear Unit, an activation function that helps the model learn non-linear patterns.
- Dropout: A regularization technique that randomly deactivates neurons during training to prevent overfitting.
- BCE Loss: Binary Cross Entropy Loss, the loss function used for binary classification problems.
- Adam Optimizer: An optimization algorithm that adjusts the model's parameters during training.
- Mini-batch: A subset of training data processed together to update the model.
- Class Weights: Values that adjust the importance of different classes during training to handle imbalanced data.
*/

# Federated Fraud Detection System Implementation

## Overview
This implementation demonstrates a federated learning approach to fraud detection, allowing multiple banks to collaboratively train a fraud detection model while keeping their transaction data private. The system uses the Flower (flwr) framework for federated learning and PyTorch for the underlying machine learning model.

## System Architecture

### 1. Data Distribution
- Each client (bank) receives a portion of the dataset
- Data split: 80% training, 20% testing for each client
- Current implementation handles 3 clients with equal data distribution

### 2. Fraud Detection Criteria
Transaction flagging uses a weighted scoring system:
```python
fraud_score = (
    2.0 * (High Transaction Amount) +    # Strong indicator
    1.5 * (Multiple Login Attempts) +    # Very important
    1.0 * (Long Transaction Duration) +  # Moderate indicator
    1.5 * (Low Account Balance)          # Important context
)
```
- Fraud threshold: 2.5 (requires multiple strong indicators)
- Results in approximately 3.90% fraud rate in the dataset

### 3. Model Architecture
The neural network model (`FraudDetector`) processes 5 key features:
1. Transaction Amount (highest weight)
2. Login Attempts
3. Account Balance
4. Transaction Duration
5. Customer Age

## Training Process

### 1. Local Training (Per Client)
- **Batch Size**: 32 transactions
- **Maximum Epochs**: 10
- **Early Stopping**:
  - Patience: 2 epochs
  - Monitors: Training loss
  - Prevents overfitting
- **Loss Function**: Binary Cross-Entropy with Logits
  - Includes class weight adjustment for imbalanced data
  - Weight calculation: `min(((1 - fraud_rate) / fraud_rate) * 0.5, 10.0)`

### 2. Federated Learning Rounds
- Server aggregates model updates from all clients
- Each round consists of:
  1. Server distributes global model
  2. Clients train locally
  3. Clients evaluate performance
  4. Server aggregates updates

## Evaluation Metrics

### 1. Threshold Selection
- Tests thresholds between 0.3 and 0.9
- Optimizes for balanced performance:
  - Minimum recall: 40%
  - Minimum precision: 10%
- Fallback mechanism if minimum requirements not met

### 2. Performance Metrics
Latest results show strong performance:

**Client 0 (Final Round)**:
- Accuracy: 93.45% → Overall correct predictions
- Balanced Accuracy: 96.65% → Equal consideration of fraud and non-fraud
- Precision: 26.67% → When flagged as fraud, correct 27% of the time
- Recall: 100% → Caught all actual fraud cases
- F1 Score: 42.11% → Balance between precision and recall
- Specificity: 93.29% → Correctly identified most legitimate transactions

**Client 2 (Final Round)**:
- Accuracy: 95.24%
- Balanced Accuracy: 83.85%
- Precision: 45.45%
- Recall: 71.43%
- F1 Score: 55.56%
- Specificity: 96.27%

## Implementation Highlights

### 1. Class Imbalance Handling
- Weighted loss function
- Dynamic threshold selection
- Balanced accuracy metrics

### 2. Early Stopping Logic
```python
if avg_epoch_loss < best_loss:
    best_loss = avg_epoch_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

### 3. Metric Calculation
- True Positives (TP): Correctly identified fraud
- False Positives (FP): Incorrectly flagged as fraud
- True Negatives (TN): Correctly identified normal transactions
- False Negatives (FN): Missed fraud cases

Derived metrics:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- Balanced Accuracy = (Recall + Specificity) / 2
- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

## Privacy Considerations
1. Data never leaves client premises
2. Only model parameters are shared
3. Each client maintains data independence
4. No raw transaction data is exposed

## Future Improvements
1. Dynamic class weight adjustment
2. More sophisticated feature engineering
3. Adaptive learning rates
4. Extended feature set
5. Anomaly detection pre-processing

## Running the System
1. Start the server:
   ```bash
   python src/server/server.py
   ```

2. Start each client (in separate terminals):
   ```bash
   python src/client/client.py --client_id [0-2] --num_clients 3 --data_path data/bank_transactions_data.csv
   ```

## Conclusion
The system demonstrates effective fraud detection while maintaining data privacy through federated learning. The balanced approach between precision and recall makes it suitable for real-world deployment, with the flexibility to adjust thresholds based on specific business requirements.