# Import necessary PyTorch modules for neural network implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector(nn.Module):
    """
    Neural Network model for fraud detection in banking transactions.
    Implements a flexible deep neural network with configurable hidden layers
    and dropout regularization for better generalization.
    """
    def __init__(self, input_dim: int = 4):
        """
        Initialize the fraud detection model.

        Args:
            input_dim (int): Number of input features from transaction data
        """
        super(FraudDetector, self).__init__()

        # Architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for better training"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.layers(x)

    def predict(self, features: Dict[str, float], threshold: float = 0.5) -> Tuple[int, float]:
        """
        Make prediction for a single transaction

        Args:
            features: Dictionary of normalized features
            threshold: Classification threshold

        Returns:
            Tuple of (prediction, probability)
        """
        # Convert features to tensor
        x = torch.tensor([
            features['amount'],
            features['balance'],
            features['age'],
            features['attempts']
        ], dtype=torch.float32).reshape(1, -1)

        # Make prediction
        with torch.no_grad():
            probability = self.forward(x).item()
            prediction = 1 if probability >= threshold else 0

        return prediction, probability

    def explain_prediction(self, features: Dict[str, float], prediction: int, probability: float) -> Dict:
        """
        Provide explanation for the model's prediction

        Args:
            features: Dictionary of normalized features
            prediction: Model's prediction (0 or 1)
            probability: Prediction probability

        Returns:
            Dictionary containing explanation and feature importance
        """
        # Calculate feature importance using gradient-based approach
        x = torch.tensor([
            features['amount'],
            features['balance'],
            features['age'],
            features['attempts']
        ], dtype=torch.float32).reshape(1, -1)
        x.requires_grad = True

        # Forward pass
        output = self.forward(x)

        # Backward pass
        output.backward()

        # Get gradients
        gradients = x.grad.abs().numpy()[0]

        # Normalize gradients to get feature importance
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
            },
            'key_factors': self._get_key_factors(features, importance)
        }

        return explanation

    def _get_key_factors(self, features: Dict[str, float], importance: np.ndarray) -> List[str]:
        """Identify key factors contributing to the prediction"""
        factors = []

        # Check transaction amount
        if abs(features['amount']) > 2.0 and importance[0] > 0.2:
            factors.append("Unusual transaction amount")

        # Check account balance
        if abs(features['balance']) > 2.0 and importance[1] > 0.2:
            factors.append("Unusual account balance")

        # Check age
        if abs(features['age']) > 2.0 and importance[2] > 0.2:
            factors.append("Age-related risk factor")

        # Check login attempts
        if features['attempts'] > 1.5 and importance[3] > 0.2:
            factors.append("Suspicious login activity")

        return factors

    def get_threshold_metrics(self, features: torch.Tensor, labels: torch.Tensor,
                            thresholds: Optional[List[float]] = None) -> Dict[float, Dict[str, float]]:
        """
        Calculate metrics for different threshold values

        Args:
            features: Input features tensor
            labels: True labels tensor
            thresholds: List of threshold values to evaluate

        Returns:
            Dictionary mapping thresholds to their metrics
        """
        if thresholds is None:
            thresholds = np.linspace(0.3, 0.9, 13)

        metrics = {}

        # Get predictions
        with torch.no_grad():
            probabilities = self.forward(features)

        # Calculate metrics for each threshold
        for threshold in thresholds:
            predictions = (probabilities >= threshold).float()

            # Calculate metrics
            tp = ((predictions == 1) & (labels == 1)).sum().item()
            tn = ((predictions == 0) & (labels == 0)).sum().item()
            fp = ((predictions == 1) & (labels == 0)).sum().item()
            fn = ((predictions == 0) & (labels == 1)).sum().item()

            # Calculate rates
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[threshold] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1
            }

        return metrics

    def get_parameters(self):
        """
        Get model parameters for federated learning.
        Converts parameters to numpy arrays for compatibility with Flower framework.

        Returns:
            list: Model parameters as numpy arrays
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Update model parameters during federated learning.

        Args:
            parameters (list): New parameter values from federated server
        """
        # Create state dict from received parameters
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)