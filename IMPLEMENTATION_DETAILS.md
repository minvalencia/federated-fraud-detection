# Fraud Detection System Implementation Details

## Architecture Overview

### 1. System Components

1. **API Layer (FastAPI)**
   - RESTful endpoints for data processing and predictions
   - Authentication and rate limiting
   - Request validation and error handling
   - Asynchronous training support

2. **ML Pipeline**
   - Data preprocessing and feature engineering
   - Model training and evaluation
   - Prediction generation
   - Model and scaler persistence

3. **Storage Layer**
   - File-based storage for uploaded data
   - Model and scaler persistence
   - Training logs and metrics

### 2. Model Architecture

The fraud detection model uses a neural network implemented in PyTorch:

```python
class FraudDetector(nn.Module):
    def __init__(self):
        super(FraudDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
```

Key features:
- 5 input features (normalized transaction attributes)
- 3 hidden layers with dropout for regularization
- Sigmoid activation for binary classification
- BCELoss for training

### 3. Training Process

#### Data Preprocessing
```python
# Features are normalized using StandardScaler
numerical_features = [
    'TransactionAmount',
    'TransactionDuration',
    'AccountBalance',
    'CustomerAge'
]
# Each feature is normalized using median and IQR
df[f'{feature}_normalized'] = (df[feature] - median) / iqr
```

#### Synthetic Fraud Label Generation
The system uses several patterns to generate synthetic fraud labels:
```python
def generate_fraud_label(row, stats):
    # Pattern 1: High amount transactions
    if transaction_amount > amount_mean * 2:
        is_fraud = True

    # Pattern 2: Amount > 80% of balance
    if transaction_amount > account_balance * 0.8:
        is_fraud = True

    # Pattern 3: Elderly customers with multiple login attempts
    if login_attempts > 3 and customer_age > 60:
        is_fraud = True

    # Pattern 4: Quick large transactions
    if transaction_amount > amount_mean * 1.5 and duration < duration_mean * 0.3:
        is_fraud = True

    # Add random noise to prevent overfitting
    if np.random.random() < 0.02:  # 2% random flip
        is_fraud = not is_fraud

    return float(is_fraud)
```

#### Training Configuration
```python
# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Training loop with early stopping
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Early stopping check
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

### 4. Prediction Process

#### Feature Processing
```python
# Normalize input features using saved scaler
X = self.scaler.transform(X)
X = torch.FloatTensor(X).to(self.device)
```

#### Dynamic Threshold Adjustment
```python
# Make predictions with dynamic threshold
with torch.no_grad():
    outputs = model(X)
    probabilities = outputs.cpu().numpy()

    # Calculate dynamic threshold adjustment based on distribution
    prob_mean = np.mean(probabilities)
    prob_std = np.std(probabilities)

    # Adjust threshold if the fraud rate would be unreasonably high
    estimated_fraud_rate = np.mean(probabilities >= threshold)
    if estimated_fraud_rate > 0.20:  # If fraud rate would be > 20%
        # Use a dynamic threshold based on distribution
        dynamic_threshold = max(
            threshold,
            min(
                prob_mean + prob_std,  # Upper bound
                0.5  # Default maximum threshold
            )
        )
```

#### Prediction Explanation
```python
def explain_prediction(self, features, prediction, probability):
    # Calculate feature importance using gradients
    gradients = x.grad.abs().numpy()
    importance = gradients / gradients.sum()

    # Create explanation
    explanation = {
        'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
        'confidence': f"{probability * 100:.2f}%",
        'feature_importance': {
            'transaction_amount': float(importance[0]),
            'account_balance': float(importance[1]),
            'age': float(importance[2]),
            'login_attempts': float(importance[3])
        }
    }
```

### 5. Performance Metrics

The system calculates various metrics for model evaluation:
```python
def calculate_metrics(predictions, true_labels):
    # Calculate basic metrics
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()

    # Calculate derived metrics
    metrics = {
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'f1_score': 2 * (precision * recall) / (precision + recall)
    }
    return metrics
```

### 6. Federated Learning Support

The system supports federated learning across multiple banks:

1. **Client Implementation**
```python
class FraudDetectionClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Update model with server parameters
        self.set_parameters(parameters)

        # Train on local data
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()

        # Return updated parameters
        return self.get_parameters(), len(X_train), metrics
```

2. **Server Configuration**
```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # 100% of clients participate
    min_fit_clients=2,
    min_available_clients=2
)
```

### 7. API Usage

1. **Upload Data**
```http
POST /upload
Content-Type: multipart/form-data
X-API-Key: your-api-token
```

2. **Process Data**
```http
POST /process/{filename}
Content-Type: application/json
X-API-Key: your-api-token
```

3. **Train Model**
```http
POST /train/{filename}
X-API-Key: your-api-token
```

4. **Make Predictions**
```http
POST /predict
Content-Type: application/json
X-API-Key: your-api-token
```

### 8. System Safeguards

1. **Early Stopping**
   - Prevents overfitting
   - Monitors validation loss
   - Implements patience mechanism

2. **Dynamic Threshold Adjustment**
   - Adapts to data distribution
   - Prevents excessive false positives
   - Maintains reasonable fraud detection rate

3. **Feature Importance Analysis**
   - Gradient-based importance calculation
   - Explains model decisions
   - Helps in feature selection

4. **Error Handling**
   - Input validation
   - Data quality checks
   - Resource management
   - Exception handling and logging

### 9. Performance Optimization

1. **Model Optimization**
   - Batch prediction support
   - CPU/GPU optimization
   - Memory management
   - Caching strategy

2. **API Optimization**
   - Async processing
   - Connection pooling
   - Response compression
   - Rate limiting

3. **Storage Optimization**
   - File cleanup
   - Data archival
   - Cache management
   - Volume monitoring