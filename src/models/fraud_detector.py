# Import necessary PyTorch modules for neural network implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FraudDetector(nn.Module):
    """
    Neural Network model for fraud detection in banking transactions.
    Implements a flexible deep neural network with configurable hidden layers
    and dropout regularization for better generalization.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        """
        Initialize the fraud detection model.

        Args:
            input_dim (int): Number of input features from transaction data
            hidden_dims (list): Dimensions of hidden layers [default: [128, 64, 32]]
                              Decreasing dimensions help in learning hierarchical features
        """
        super(FraudDetector, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer: Takes transaction features and maps to first hidden dimension
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers: Create a funnel architecture for feature extraction
        # Each layer reduces dimension, helping to distill important fraud indicators
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        # Output layer: Single node with sigmoid activation for binary classification
        # Output will be probability of transaction being fraudulent
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Dropout layer: Prevents overfitting by randomly deactivating 30% of neurons
        # This improves model generalization and robustness
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input transaction data batch

        Returns:
            torch.Tensor: Probability of fraud for each transaction
        """
        # Process through hidden layers with ReLU activation and dropout
        for layer in self.layers:
            x = F.relu(layer(x))  # ReLU helps with vanishing gradient problem
            x = self.dropout(x)   # Apply dropout for regularization

        # Final layer with sigmoid activation to get probability
        x = torch.sigmoid(self.output_layer(x))
        return x

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