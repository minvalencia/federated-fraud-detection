# Federated Learning Details

/*
Key Terms Explained:
- Federated Learning: A machine learning approach where multiple parties train models locally on their private data, sharing only model updates without exposing raw data.
- Global Server: The central coordinator that aggregates model updates from all clients and distributes the improved model.
- Client: A participating bank or institution that trains the model on their local data.
- Model Parameters: The weights and biases of the neural network that are shared between clients and server.
- Aggregation: The process of combining model updates from multiple clients into a single improved model.
- Round: One complete cycle of local training across all clients followed by model aggregation.
*/

# Federated Learning Implementation Details

## System Overview

### 1. Architecture Components
```
Server (Coordinator)
    │
    ├── Client 0 (Bank A)
    ├── Client 1 (Bank B)
    └── Client 2 (Bank C)
```

Each component maintains:
- Server: Global model state
- Clients: Local data and temporary model copy

### 2. Communication Flow
1. Server → Clients: Global model parameters
2. Clients: Local training
3. Clients → Server: Updated parameters
4. Server: Parameter aggregation
5. Repeat for N rounds

## Client Implementation

### 1. NumPyClient Interface
```python
class FraudDetectionClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        # Local training
        return parameters, num_samples, metrics

    def evaluate(self, parameters, config):
        # Local evaluation
        return accuracy, num_samples, metrics
```

### 2. Parameter Handling
- **Getting Parameters**:
  ```python
  def get_parameters(self):
      return [val.cpu().numpy() for _, val in self.state_dict().items()]
  ```

- **Setting Parameters**:
  ```python
  def set_parameters(self, parameters):
      params_dict = zip(self.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      self.load_state_dict(state_dict, strict=True)
  ```

## Training Process

### 1. Local Training (Per Client)
```python
def fit(self, parameters, config):
    self.set_parameters(parameters)

    for epoch in range(10):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Early stopping logic
        if early_stopping_condition:
            break

    return self.get_parameters(), len(X_train), {"loss": best_loss}
```

### 2. Privacy Preservation
- Only model parameters are shared
- Raw data never leaves client
- No direct access between clients
- Aggregated updates only

## Server Configuration

### 1. Server Setup
```python
def start_server(self, host: str = "0.0.0.0", port: int = 8081):
    server = Server(
        client_manager=SimpleClientManager(),
        strategy=FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2
        )
    )
```

### 2. Strategy Parameters
- **fraction_fit**: 1.0 (use all available clients)
- **min_fit_clients**: 2 (minimum clients for training)
- **min_evaluate_clients**: 2 (minimum clients for evaluation)
- **min_available_clients**: 2 (minimum before starting round)

## Data Management

### 1. Data Distribution
```python
def load_and_preprocess_data(data_path, client_id, num_clients):
    # Calculate client's data portion
    total_samples = len(X_scaled)
    samples_per_client = total_samples // num_clients
    start_idx = client_id * samples_per_client
    end_idx = start_idx + samples_per_client
```

### 2. Local Data Handling
- Independent standardization per client
- Local train/test split (80/20)
- Client-specific fraud labeling

## Performance Monitoring

### 1. Metrics Tracked
Per client, per round:
- Training loss
- Evaluation metrics
- Model convergence
- Client participation

### 2. Convergence Criteria
- Early stopping per client
- Global model aggregation
- Performance thresholds

## Security Considerations

### 1. Data Protection
- No raw data transmission
- Encrypted parameter updates
- Independent data processing

### 2. System Security
- Secure channel (TLS/SSL)
- Client authentication
- Rate limiting
- Error handling

## Deployment Guide

### 1. Prerequisites
- Python 3.8+
- PyTorch
- Flower (flwr)
- Network connectivity

### 2. Starting System
1. Launch server:
   ```bash
   python src/server/server.py
   ```

2. Start clients:
   ```bash
   python src/client/client.py --client_id [0-2] \
                              --num_clients 3 \
                              --data_path data/bank_transactions_data.csv
   ```

### 3. Monitoring
- Watch server logs for round progress
- Monitor client metrics
- Check convergence status

## Troubleshooting

### 1. Common Issues
- Connection timeouts
- Memory constraints
- GPU availability
- Parameter synchronization

### 2. Solutions
- Retry mechanisms
- Batch size adjustment
- CPU fallback
- Logging and debugging

## Future Enhancements

### 1. Technical Improvements
- Differential privacy
- Adaptive aggregation
- Dynamic client selection
- Compression techniques

### 2. Operational Improvements
- Automated deployment
- Performance optimization
- Scaling capabilities
- Recovery mechanisms