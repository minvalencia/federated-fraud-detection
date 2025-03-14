# Federated Learning Process

/*
Key Terms Explained:
- Training Round: A complete cycle of distributed training, including local updates and global aggregation.
- Early Stopping: A technique to prevent overtraining by monitoring performance and stopping when improvements plateau.
- Parameter Encryption: The process of securing model updates during transmission between clients and server.
- Batch Size: The number of training examples processed together in one forward/backward pass.
- Learning Rate: Controls how much the model adjusts its parameters during training.
- Convergence: The process of the model reaching a stable and optimal performance level.
- FedAvg: The federated averaging algorithm used to combine model updates from multiple clients.
*/

# Federated Learning Process Deep Dive

## 1. System Architecture

### Server-Client Communication
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

### Data Privacy
- Each bank keeps its transaction data
- Only model updates are shared
- No raw data exposure
- Secure parameter transmission

## 2. Training Rounds Explained

### Round Structure
```python
# Round 1
Server → Client A: Initial model parameters
Client A: Trains for 10 epochs or until early stopping
Client A → Server: Updated parameters, metrics

Server → Client B: Same initial model
Client B: Trains independently
Client B → Server: Its updates, metrics

Server: Aggregates updates using FedAvg
```

### Example Round Metrics
```
Round 1:
Client A:
- Initial Loss: 1.0537
- Final Loss: 0.8416
- Early Stop: No (completed 10 epochs)

Client B:
- Initial Loss: 0.8446
- Final Loss: 0.8076
- Early Stop: Yes (epoch 7)

Global Model:
- Aggregated parameters
- Average improvement: 20.1%
```

## 3. Parameter Handling

### Getting Parameters
```python
def get_parameters(self):
    """Convert PyTorch parameters to NumPy"""
    return [
        val.cpu().numpy()  # Convert to CPU and NumPy
        for _, val in self.state_dict().items()
    ]
```

### Setting Parameters
```python
def set_parameters(self, parameters):
    """Update model with new parameters"""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({
        k: torch.tensor(v)
        for k, v in params_dict
    })
    self.load_state_dict(state_dict, strict=True)
```

## 4. Training Process Details

### Local Training Loop
```python
def fit(self, parameters, config):
    # Setup
    self.set_parameters(parameters)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Early stopping check
        avg_loss = epoch_loss / len(train_loader)
        if self._should_stop_early(avg_loss):
            break

    return self.get_parameters(), len(X_train), metrics
```

## 5. Performance Monitoring

### Client Metrics
```
Client 0 (Final Results):
- Accuracy: 93.45%
- Balanced Accuracy: 96.65%
- Precision: 26.67%
- Recall: 100%
- F1 Score: 42.11%

Client 2 (Final Results):
- Accuracy: 95.24%
- Balanced Accuracy: 83.85%
- Precision: 45.45%
- Recall: 71.43%
- F1 Score: 55.56%
```

### Convergence Analysis
```
Training Progress:
Round 1: High variance between clients
Round 2: More consistent performance
Round 3: Stabilized metrics
```

## 6. Security Implementation

### Parameter Encryption
```python
# Example of secure parameter transmission
def secure_transmission(parameters):
    encrypted = encrypt_parameters(parameters)
    signature = sign_parameters(parameters)
    return {
        'data': encrypted,
        'signature': signature,
        'timestamp': current_time()
    }
```

### Authentication Flow
```
1. Client connects to server
2. Server verifies client credentials
3. Establish secure channel (TLS)
4. Exchange encrypted parameters
5. Verify signatures
```

## 7. Error Handling

### Common Scenarios
```python
try:
    # Training attempt
    parameters = train_local_model()
except ConnectionError:
    # Handle connection loss
    cache_parameters()
    retry_connection()
except MemoryError:
    # Handle resource constraints
    reduce_batch_size()
    clear_memory()
    restart_training()
```

## 8. Deployment Process

### Server Setup
```bash
# 1. Start server
python src/server/server.py

# Server output:
INFO: Server started (port: 8081)
INFO: Waiting for clients...
INFO: Client connected (id: 0)
INFO: Starting round 1...
```

### Client Setup
```bash
# Start Client 0
python src/client/client.py \
    --client_id 0 \
    --num_clients 3 \
    --data_path data/bank_transactions_data.csv

# Client output:
INFO: Connected to server
INFO: Received initial parameters
INFO: Starting local training...
```

## 9. Monitoring and Debugging

### Log Analysis
```python
# Example log format
{
    'timestamp': '2025-03-15 00:03:37,877',
    'client_id': 0,
    'round': 1,
    'epoch': 3,
    'loss': 0.8750,
    'metrics': {
        'accuracy': 0.9345,
        'precision': 0.2667,
        'recall': 1.0000
    }
}
```

### Performance Tracking
```python
def monitor_performance():
    track_metrics = {
        'loss_trend': [],
        'accuracy_trend': [],
        'convergence_rate': [],
        'client_participation': set()
    }
```

## 10. Future Enhancements

### Planned Features
1. **Adaptive Aggregation**:
   ```python
   def adaptive_fedavg(updates):
       weights = calculate_client_weights(updates)
       return weighted_aggregate(updates, weights)
   ```

2. **Dynamic Client Selection**:
   ```python
   def select_clients(round):
       performance = get_client_history()
       return optimize_selection(performance)
   ```

3. **Advanced Privacy**:
   ```python
   def differential_privacy():
       noise = generate_gaussian_noise()
       return add_noise_to_parameters(noise)
   ```