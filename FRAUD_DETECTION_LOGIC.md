# Fraud Detection Logic

/*
Key Terms Explained:
- Feature Weights: Numerical values assigned to different transaction characteristics to determine their importance in fraud detection.
- Percentile Thresholds: Statistical cutoff points (e.g., 90th percentile) used to identify unusual transaction patterns.
- Score Calculation: The mathematical process of combining multiple weighted features to produce a final fraud risk score.
- False Positive: A legitimate transaction incorrectly flagged as fraudulent.
- False Negative: A fraudulent transaction incorrectly classified as legitimate.
- Precision: The percentage of correctly identified fraud cases among all transactions flagged as fraud.
- Recall: The percentage of actual fraud cases that were successfully detected.
- F1 Score: A balanced measure combining precision and recall.
*/

## Transaction Scoring System

### 1. Feature Weights and Rationale
```python
fraud_score = (
    2.0 * (TransactionAmount > 90th_percentile) +  # Weight: 2.0
    1.5 * (LoginAttempts > 2) +                    # Weight: 1.5
    1.0 * (TransactionDuration > 90th_percentile) + # Weight: 1.0
    1.5 * (AccountBalance < 10th_percentile)        # Weight: 1.5
)
```

#### Weight Justification:
- **Transaction Amount (2.0)**: Highest weight because unusually large transactions are the strongest fraud indicators
- **Login Attempts (1.5)**: Multiple failed logins often indicate attempted unauthorized access
- **Account Balance (1.5)**: Low balance combined with other factors may indicate account takeover
- **Transaction Duration (1.0)**: Longer duration might indicate automated/scripted attacks

### 2. Threshold Calculation
- **Fraud Threshold = 2.5**
  - Requires at least two strong indicators or three moderate indicators
  - Examples of fraud combinations:
    1. High amount (2.0) + Multiple logins (1.5) = 3.5 > 2.5
    2. Low balance (1.5) + Multiple logins (1.5) = 3.0 > 2.5
    3. High duration (1.0) + High amount (2.0) = 3.0 > 2.5

## Model Architecture

### 1. Neural Network Structure
```python
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
```

- **Input Layer**: 5 features (normalized)
- **Hidden Layer 1**: 32 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation
- **Output Layer**: Single neuron (fraud probability)
- **Dropout**: 20% dropout for regularization

### 2. Feature Processing
1. **Standardization**:
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
   - Centers features around 0
   - Scales to unit variance
   - Helps with model convergence

2. **Feature Importance Order**:
   ```python
   feature_columns = [
       'TransactionAmount',    # Primary indicator
       'LoginAttempts',        # Security indicator
       'AccountBalance',       # Context indicator
       'TransactionDuration',  # Behavioral indicator
       'CustomerAge'          # Risk factor
   ]
   ```

## Training Process Details

### 1. Class Imbalance Handling
```python
pos_weight = min(((1 - y_train.mean()) / y_train.mean()) * 0.5, 10.0)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

- **Weight Calculation Example**:
  - If fraud_rate = 3.90%
  - Initial weight = (1 - 0.039) / 0.039 = 24.64
  - Dampened weight = 24.64 * 0.5 = 12.32
  - Final weight = min(12.32, 10.0) = 10.0

### 2. Batch Processing
```python
batch_size = 32
dataset = TensorDataset(X_train, y_train.view(-1, 1))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

- **Batch Size Rationale**:
  - Small enough for good generalization
  - Large enough for stable gradients
  - Shuffled to prevent learning order dependencies

## Evaluation Strategy

### 1. Dynamic Threshold Selection
```python
thresholds = torch.linspace(0.3, 0.9, 13)
```

- **Range Justification**:
  - Lower bound (0.3): Captures high recall
  - Upper bound (0.9): Ensures high precision
  - 13 steps: Fine-grained threshold optimization

### 2. Scoring System
```python
if recall >= 0.4 and precision >= 0.1:
    score = balanced_acc + precision + recall
```

- **Minimum Requirements**:
  - 40% recall: Catch reasonable amount of fraud
  - 10% precision: Limit false positives
  - Balanced accuracy included for overall performance

### 3. Performance Interpretation
Current Results Analysis:
- **Client 0**: High recall (100%) with moderate precision (26.67%)
  - Good for catching all fraud
  - Higher false positive rate acceptable
- **Client 2**: Balanced performance (71.43% recall, 45.45% precision)
  - Better precision but misses some fraud
  - Lower false positive rate

## Future Optimizations

### 1. Immediate Improvements
- Adaptive class weights based on validation performance
- Feature interaction terms
- Regular threshold recalibration

### 2. Advanced Enhancements
- Sequence modeling for temporal patterns
- Attention mechanisms for feature importance
- Ensemble methods with different architectures

### 3. Production Considerations
- Model versioning and rollback capability
- Performance monitoring and alerts
- Periodic model retraining schedule