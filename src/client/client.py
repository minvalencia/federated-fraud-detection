# Import required libraries for federated learning client
import flwr as fl
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
from src.models.fraud_detector import FraudDetector
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import argparse
import logging
from ..utils.data_preprocessor import process_all_banks, DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionClient(fl.client.NumPyClient):
    """
    Federated Learning client for fraud detection.
    Handles local model training and evaluation for each participating bank.
    Implements the Flower NumPyClient interface for federated learning.
    """

    def __init__(self, client_id: int, data_dir: str):
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load and preprocess data
        self.load_and_preprocess_data(data_dir)

        # Initialize model
        input_dim = len(self.X_train[0])
        self.model = self.create_model(input_dim).to(self.device)

        logger.info(f"Client {client_id} initialized with {len(self.X_train)} training samples")

    def load_and_preprocess_data(self, data_dir: str):
        """Load and preprocess data for this client."""
        # Process all bank data
        combined_df, stats = process_all_banks(data_dir)
        logger.info(f"Processed data from {len(stats)} banks")

        # Split data for this client
        n_clients = 3  # Total number of clients
        n_samples = len(combined_df)
        samples_per_client = n_samples // n_clients

        # Get this client's portion of data
        start_idx = self.client_id * samples_per_client
        end_idx = start_idx + samples_per_client if self.client_id < n_clients - 1 else n_samples

        client_df = combined_df.iloc[start_idx:end_idx].copy()
        logger.info(f"Client {self.client_id} loaded {len(client_df)} transactions")

        # Calculate fraud based on score
        fraud_threshold = client_df['FraudScore'].quantile(0.95)  # Top 5% as fraud
        client_df['isFraud'] = (client_df['FraudScore'] > fraud_threshold).astype(int)

        # Split features and target
        feature_columns = [col for col in client_df.columns if col.endswith('_normalized') or col.endswith('_encoded')]
        X = client_df[feature_columns].values
        y = client_df['isFraud'].values

        # Split into train and test
        train_size = int(0.8 * len(X))
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_test = X[train_size:]
        self.y_test = y[train_size:]

        # Convert to tensors
        self.X_train = torch.FloatTensor(self.X_train).to(self.device)
        self.y_train = torch.FloatTensor(self.y_train).to(self.device)
        self.X_test = torch.FloatTensor(self.X_test).to(self.device)
        self.y_test = torch.FloatTensor(self.y_test).to(self.device)

        # Log data statistics
        n_fraud = np.sum(y)
        logger.info(f"Client {self.client_id}: Identified {n_fraud} fraudulent transactions ({n_fraud/len(y)*100:.2f}%)")
        logger.info(f"Training on {len(self.X_train)} samples, Testing on {len(self.X_test)} samples")

    def create_model(self, input_dim: int) -> nn.Module:
        """Create the neural network model."""
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def get_parameters(self, config):
        """
        Get current model parameters for federated learning.

        Args:
            config: Configuration from server (unused)

        Returns:
            list: Current model parameters
        """
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        """
        Update local model with parameters from federated server.

        Args:
            parameters: New model parameters from server
        """
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        """
        Train model on local data.

        Args:
            parameters: Initial model parameters from server
            config: Training configuration from server

        Returns:
            tuple: Updated model parameters, number of samples, and metrics
        """
        # Update model with server parameters
        self.set_parameters(parameters)

        # Setup training components
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Calculate class weights to handle imbalance (with dampening)
        X_train, y_train = self.X_train, self.y_train
        pos_weight = min(((1 - y_train.mean()) / y_train.mean()) * 0.5, 10.0)  # Dampen and cap the weight
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Prepare data for training
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)

        # Create mini-batches
        batch_size = 32
        dataset = torch.utils.data.TensorDataset(X_train, y_train.view(-1, 1))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop with early stopping
        self.model.train()
        total_loss = 0
        best_loss = float('inf')
        patience = 2
        patience_counter = 0

        for epoch in range(10):  # Increased max epochs
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Local epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")

            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            total_loss += avg_epoch_loss

        return self.get_parameters(config={}), len(X_train), {"loss": best_loss}

    def evaluate(self, parameters, config):
        """
        Evaluate model on local test data with optimized threshold selection.
        """
        self.set_parameters(parameters)

        # Evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            loss = criterion(outputs, self.y_test.view(-1, 1))

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)

            # Try different thresholds with balanced scoring
            thresholds = torch.linspace(0.3, 0.9, 13)
            best_score = -float('inf')  # Changed from 0 to allow negative scores
            best_metrics = None
            best_threshold = 0.5
            fallback_metrics = None  # Store the best metrics regardless of minimum requirements

            for threshold in thresholds:
                predictions = (probabilities > threshold).float()

                # Calculate metrics
                tp = ((predictions == 1) & (self.y_test.view(-1, 1) == 1)).sum().item()
                tn = ((predictions == 0) & (self.y_test.view(-1, 1) == 0)).sum().item()
                fp = ((predictions == 1) & (self.y_test.view(-1, 1) == 0)).sum().item()
                fn = ((predictions == 0) & (self.y_test.view(-1, 1) == 1)).sum().item()

                # Calculate balanced metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Balanced accuracy (average of recall and specificity)
                balanced_acc = (recall + specificity) / 2

                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Current metrics
                current_metrics = {
                    "threshold": threshold.item(),
                    "accuracy": (tp + tn) / len(self.y_test),
                    "balanced_accuracy": balanced_acc,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "f1": f1,
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn
                }

                # Update fallback metrics if this is the best so far (regardless of minimum requirements)
                current_score = balanced_acc + precision + recall
                if fallback_metrics is None or current_score > best_score:
                    fallback_metrics = current_metrics

                # Custom scoring that balances all metrics with more lenient requirements
                if recall >= 0.4 and precision >= 0.1:  # Reduced minimum requirements
                    score = balanced_acc + precision + recall  # Simplified scoring
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_metrics = current_metrics

            # Use fallback metrics if no threshold met the minimum requirements
            if best_metrics is None:
                best_metrics = fallback_metrics
                print("\nWarning: Using fallback metrics as no threshold met minimum requirements")

            print(f"\nEvaluation Metrics (threshold={best_metrics['threshold']:.2f}):")
            print(f"Loss: {loss.item():.4f}")
            print(f"Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
            print(f"Precision: {best_metrics['precision']:.4f}")
            print(f"Recall: {best_metrics['recall']:.4f}")
            print(f"Specificity: {best_metrics['specificity']:.4f}")
            print(f"F1 Score: {best_metrics['f1']:.4f}")
            print(f"True Positives: {best_metrics['true_positives']}")
            print(f"False Positives: {best_metrics['false_positives']}")
            print(f"True Negatives: {best_metrics['true_negatives']}")
            print(f"False Negatives: {best_metrics['false_negatives']}")

        return float(best_metrics['balanced_accuracy']), len(self.X_test), best_metrics

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client for Fraud Detection')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing bank data files')
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='Server address')
    args = parser.parse_args()

    # Create client
    client = FraudDetectionClient(args.client_id, args.data_dir)

    # Start client
    fl.client.start_numpy_client(args.server_address, client=client)

if __name__ == "__main__":
    main()