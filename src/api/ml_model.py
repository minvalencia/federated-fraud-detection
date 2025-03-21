import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class FraudDetector(nn.Module):
    def __init__(self, input_dim: int):
        super(FraudDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.scaler_path = self.model_dir / "scaler.joblib"
        self.model_path = self.model_dir / "fraud_detector.pt"

        # Initialize model and scaler
        self.input_dim = 5  # Number of features after preprocessing
        self.model = FraudDetector(self.input_dim).to(self.device)
        self.scaler = None

        # Expected feature names and their valid ranges
        self.expected_features = [
            'amount_normalized',
            'duration_normalized',
            'attempts_normalized',
            'balance_normalized',
            'age_normalized'
        ]

        self.valid_ranges = {
            'amount_normalized': (-5, 5),
            'duration_normalized': (-5, 5),
            'attempts_normalized': (-5, 5),
            'balance_normalized': (-5, 5),
            'age_normalized': (-5, 5)
        }

        # Load model and scaler if they exist
        self.load_model()

    def load_model(self):
        """Load the saved model and scaler"""
        try:
            if self.model_path.exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info("Loaded existing model")
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing scaler")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def save_model(self):
        """Save the current model and scaler"""
        try:
            torch.save(self.model.state_dict(), self.model_path)
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
            logger.info("Saved model and scaler")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> Dict:
        """Train the model on new data"""
        self.model.train()

        # Convert to tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        # Calculate class weights
        pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Save updated model
        self.save_model()

        return {"losses": losses}

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

            # Check value ranges
            for i, feature in enumerate(self.expected_features):
                min_val, max_val = self.valid_ranges[feature]
                if not ((X[:, i] >= min_val) & (X[:, i] <= max_val)).all():
                    return False, f"Values out of range for feature {feature}"

            return True, "Validation successful"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def predict(self, X: np.ndarray, threshold: float = 0.5, feature_names: List[str] = None) -> Dict:
        """Make predictions on new data with validation and detailed summary"""
        self.model.eval()

        # Validate input features
        is_valid, message = self.validate_features(X, feature_names)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {message}")

        # Convert to tensor
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)

        # Generate detailed prediction summary
        n_samples = len(predictions)
        n_fraud = np.sum(predictions)
        n_legitimate = n_samples - n_fraud

        # Get indices of fraud predictions
        fraud_indices = np.where(predictions == 1)[0]

        # Get fraud details
        fraud_details = []
        for idx in fraud_indices:
            fraud_details.append({
                "index": int(idx),
                "probability": float(probabilities[idx][0]),
                "features": {
                    name: float(X[idx][i].cpu().numpy())
                    for i, name in enumerate(self.expected_features)
                } if feature_names else None
            })

        # Sort fraud details by probability (highest first)
        fraud_details.sort(key=lambda x: x["probability"], reverse=True)

        # Calculate probability distribution statistics
        prob_stats = {
            "mean": float(np.mean(probabilities)),
            "median": float(np.median(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities))
        }

        # Calculate prediction confidence
        fraud_probs = probabilities[predictions == 1]
        legitimate_probs = probabilities[predictions == 0]

        confidence_stats = {
            "avg_fraud_confidence": float(np.mean(fraud_probs)) if len(fraud_probs) > 0 else 0,
            "avg_legitimate_confidence": float(1 - np.mean(legitimate_probs)) if len(legitimate_probs) > 0 else 0
        }

        return {
            "predictions": predictions.flatten(),
            "probabilities": probabilities.flatten(),
            "fraud_details": fraud_details,
            "summary": {
                "total_transactions": n_samples,
                "fraud_detected": int(n_fraud),
                "legitimate_transactions": int(n_legitimate),
                "fraud_percentage": float(n_fraud/n_samples * 100),
                "probability_stats": prob_stats,
                "confidence_stats": confidence_stats,
                "threshold_used": threshold
            }
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X)

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
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }

# Global model manager instance
model_manager = ModelManager()