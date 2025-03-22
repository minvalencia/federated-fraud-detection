import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, Tuple, List
import os
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self):
        """Initialize the fraud detection model."""
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

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

    def save(self, path):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model from disk."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class ModelManager:
    def __init__(self):
        """Initialize the model manager."""
        # Set device
        self.device = torch.device('cpu')

        # Model paths
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / 'fraud_detector.pt'
        self.scaler_path = self.model_dir / 'scaler.joblib'

        # Initialize scaler
        self.scaler = None
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(str(self.scaler_path))  # Convert path to string
                logger.info("Loaded existing scaler successfully")
            except Exception as e:
                logger.error(f"Error loading scaler from {str(self.scaler_path)}: {str(e)}")
                logger.warning("Will create new scaler during training.")
        else:
            logger.info("No existing scaler found. Will create during training.")

        # Feature configuration
        self.input_dim = 5
        self.expected_features = [
            'amount_normalized',
            'duration_normalized',
            'attempts_normalized',
            'balance_normalized',
            'age_normalized'
        ]

        # Feature ranges for validation
        self.valid_ranges = {
            'amount_normalized': (-5.0, 5.0),
            'duration_normalized': (-5.0, 5.0),
            'attempts_normalized': (-5.0, 5.0),
            'balance_normalized': (-5.0, 5.0),
            'age_normalized': (-5.0, 5.0)
        }

        # Feature importance weights
        self.feature_weights = {
            'amount_normalized': 2.0,    # High importance for transaction amount
            'duration_normalized': 1.2,  # Medium importance for duration
            'attempts_normalized': 1.8,  # High importance for login attempts
            'balance_normalized': 1.5,   # Medium-high importance for balance
            'age_normalized': 1.0        # Base importance for age
        }

        # Initialize model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the fraud detection model."""
        try:
            # Create new model instance
            self.model = FraudDetector()

            # Load existing model if available
            if self.model_path.exists():
                try:
                    self.model.load(self.model_path)
                    logger.info("Loaded existing model successfully")
                except Exception as e:
                    logger.warning(f"Could not load existing model: {str(e)}. Initializing new model.")

            # Initialize weights if new model
            if not self.model_path.exists():
                # Apply feature importance weights
                for feat_name, weight in self.feature_weights.items():
                    idx = self.expected_features.index(feat_name)
                    self.model.model[0].weight.data[:, idx] *= weight

                # Initialize final layers
                nn.init.xavier_normal_(self.model.model[-2].weight, gain=2.5)
                self.model.model[-2].bias.data.fill_(-2.5)

                logger.info("Initialized new model with custom weights")

            # Ensure model is on correct device
            self.model.model.to(self.device)

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> Dict:
        """Train the fraud detection model."""
        try:
            # Initialize and fit scaler if not exists
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
                # Save scaler with explicit string path
                try:
                    joblib.dump(self.scaler, str(self.scaler_path))
                    logger.info(f"Created and saved new scaler to {str(self.scaler_path)}")
                except Exception as e:
                    logger.error(f"Failed to save scaler to {str(self.scaler_path)}: {str(e)}")
                    raise
            else:
                X = self.scaler.transform(X)

            # Convert to tensors
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

            # Setup training
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.model.parameters(), lr=0.001)

            # Training loop
            losses = []
            for epoch in range(epochs):
                # Forward pass
                self.model.model.train()
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

            # Save model
            self.save_model()

            # Verify scaler was saved
            if not self.scaler_path.exists():
                raise ValueError(f"Scaler file not found at {str(self.scaler_path)} after training")

            return {
                "losses": losses,
                "final_loss": losses[-1]
            }

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Dict:
        """Make fraud predictions with dynamic threshold adjustment."""
        try:
            # Check if scaler exists and can be loaded
            if self.scaler is None:
                if not self.scaler_path.exists():
                    raise ValueError(f"No scaler found at {str(self.scaler_path)}. Model needs to be trained first.")
                try:
                    self.scaler = joblib.load(str(self.scaler_path))
                except Exception as e:
                    raise ValueError(f"Failed to load scaler from {str(self.scaler_path)}: {str(e)}")

            # Transform features
            X = self.scaler.transform(X)

            # Convert to tensor
            X = torch.FloatTensor(X).to(self.device)

            # Make predictions
            self.model.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = outputs.cpu().numpy().flatten()

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
                    logger.info(f"Adjusting threshold from {threshold} to {dynamic_threshold} due to high fraud rate")
                    threshold = dynamic_threshold

                # Apply threshold and get predictions
                predictions = (probabilities >= threshold).astype(int)

                # Additional validation
                fraud_rate = np.mean(predictions)
                if fraud_rate > 0.20:  # If still too high, apply stricter threshold
                    percentile_threshold = np.percentile(probabilities, 80)  # Top 20%
                    threshold = max(threshold, percentile_threshold)
                    predictions = (probabilities >= threshold).astype(int)
                    logger.warning(f"Applied stricter threshold {threshold} to reduce fraud rate")

            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "adjusted_threshold": threshold
            }

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def save_model(self):
        """Save the current model."""
        try:
            self.model.save(self.model_path)
            logger.info("Saved model successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def validate_features(self, X: np.ndarray, feature_names: List[str] = None) -> Tuple[bool, str]:
        """Validate input features"""
        try:
            # Check number of features
            if X.shape[1] != self.input_dim:
                return False, f"Expected {self.input_dim} features, got {X.shape[1]}"

            # Check feature names if provided
            if feature_names:
                missing_features = set(self.expected_features) - set(feature_names)
                if missing_features:
                    return False, f"Missing required features: {missing_features}"

                # Check feature order
                if feature_names != self.expected_features:
                    return False, "Features are not in the expected order"

            # Check for NaN or infinite values
            if not np.isfinite(X).all():
                return False, "Input contains NaN or infinite values"

            # Check value ranges with more lenient validation
            for i, feature in enumerate(self.expected_features):
                min_val, max_val = self.valid_ranges[feature]
                feature_values = X[:, i]
                if np.any(feature_values < min_val * 2) or np.any(feature_values > max_val * 2):
                    return False, f"Values extremely out of range for feature {feature}"

            return True, "Validation successful"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        try:
            result = self.predict(X)
            predictions = result["predictions"]

            # Calculate metrics
            tp = np.sum((predictions == 1) & (y == 1))
            tn = np.sum((predictions == 0) & (y == 0))
            fp = np.sum((predictions == 1) & (y == 0))
            fn = np.sum((predictions == 0) & (y == 1))

            # Calculate performance metrics
            accuracy = (tp + tn) / len(y)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

# Global model manager instance
model_manager = ModelManager()