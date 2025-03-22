from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import shutil
import os
import secrets
from .ml_model import model_manager, FraudDetector
import asyncio
import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate API token if not exists
API_TOKEN_FILE = Path("data/api_token.txt")
if not API_TOKEN_FILE.exists():
    API_TOKEN = secrets.token_urlsafe(32)
    API_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_TOKEN_FILE.write_text(API_TOKEN)
else:
    API_TOKEN = API_TOKEN_FILE.read_text().strip()

# Initialize training status
training_status = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 50,
    "current_loss": 0,
    "best_loss": float('inf'),
    "error": None,
    "message": "Not started",
    "start_time": None,
    "end_time": None
}

async def verify_token(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or x_api_key != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return x_api_key

app = FastAPI(
    title="Fraud Detection Data Processor",
    description="API for processing bank transaction data for fraud detection",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if it doesn't exist
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create models directory if it doesn't exist
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class ColumnMapper:
    """Maps various column names to standardized format"""

    STANDARD_COLUMNS = {
        'TransactionAmount': [
            'transaction_amount', 'amount', 'transaction_value',
            'value', 'amt', 'transaction_amt'
        ],
        'TransactionDuration': [
            'duration', 'transaction_duration', 'time_taken',
            'processing_time', 'duration_seconds'
        ],
        'LoginAttempts': [
            'login_attempts', 'attempts', 'num_attempts',
            'login_tries', 'authentication_attempts'
        ],
        'AccountBalance': [
            'balance', 'account_balance', 'current_balance',
            'available_balance', 'bal'
        ],
        'CustomerAge': [
            'age', 'customer_age', 'client_age',
            'account_holder_age', 'holder_age'
        ],
        'TransactionType': [
            'type', 'transaction_type', 'tx_type',
            'payment_type', 'transaction_category'
        ],
        'Channel': [
            'channel', 'transaction_channel', 'payment_channel',
            'medium', 'transaction_medium'
        ],
        'CustomerOccupation': [
            'occupation', 'customer_occupation', 'profession',
            'job', 'employment'
        ]
    }

    @classmethod
    def guess_column_mapping(cls, df: pd.DataFrame) -> Dict[str, str]:
        """
        Guess the mapping between provided columns and standard columns.
        Uses fuzzy matching and common patterns.
        """
        mapping = {}
        df_columns_lower = [col.lower().replace('_', '').replace(' ', '') for col in df.columns]

        for standard_col, variations in cls.STANDARD_COLUMNS.items():
            # Add standardized version to variations
            variations = [v.lower().replace('_', '').replace(' ', '') for v in variations]
            variations.append(standard_col.lower().replace('_', '').replace(' ', ''))

            # Find best match
            for idx, col in enumerate(df_columns_lower):
                if col in variations:
                    mapping[standard_col] = df.columns[idx]
                    break

        return mapping

    @classmethod
    def validate_numeric_columns(cls, df: pd.DataFrame, mapping: Dict[str, str]) -> List[str]:
        """Validate that mapped numeric columns contain valid numeric data"""
        numeric_columns = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts',
                         'AccountBalance', 'CustomerAge']

        invalid_columns = []
        for col in numeric_columns:
            if col in mapping:
                try:
                    pd.to_numeric(df[mapping[col]])
                except:
                    invalid_columns.append(col)

        return invalid_columns

@app.get("/token", include_in_schema=False)
async def get_token():
    """Get the API token - Only available during development"""
    return {"token": API_TOKEN}

@app.post("/upload")
async def upload_file(file: UploadFile = File(..., description="CSV file containing transaction data")):
    """
    Upload a CSV file containing transaction data.

    The file should be a CSV with the following columns (or similar names):
    - TransactionAmount: Amount of the transaction
    - TransactionDuration: Duration of the transaction in seconds
    - LoginAttempts: Number of login attempts
    - AccountBalance: Current account balance
    - CustomerAge: Age of the customer
    - TransactionType: Type of transaction (e.g., transfer, payment)
    - Channel: Transaction channel (e.g., online, mobile)

    Returns:
        dict: Information about the uploaded file including:
            - filename: Name of the saved file
            - columns: List of detected columns
            - rows: Number of rows in the file
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )

    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bank_data_{timestamp}.csv"
        file_path = UPLOAD_DIR / filename

        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read and validate the file
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("The CSV file is empty")

            return {
                "filename": filename,
                "columns": df.columns.tolist(),
                "rows": len(df),
                "message": "File uploaded successfully"
            }
        except Exception as e:
            # Clean up the file if there's an error
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Error processing CSV file: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@app.get("/validate/{filename}")
async def validate_mapping(filename: str, token: str = Depends(verify_token)):
    """
    Validate the uploaded file and automatically map columns.

    Args:
        filename: Name of the uploaded CSV file

    Returns:
        dict: Validation results including:
            - mapped_columns: Automatically detected column mappings
            - numeric_validation: Results of numeric column validation
            - data_stats: Basic statistics of the data
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File {filename} not found"
            )

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Automatically guess column mapping
        mapping = ColumnMapper.guess_column_mapping(df)

        # Validate numeric columns
        invalid_columns = ColumnMapper.validate_numeric_columns(df, mapping)

        # Calculate basic statistics for numeric columns
        stats = {}
        for col_name, mapped_col in mapping.items():
            if col_name in ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'CustomerAge']:
                try:
                    col_stats = {
                        'mean': float(df[mapped_col].mean()),
                        'min': float(df[mapped_col].min()),
                        'max': float(df[mapped_col].max()),
                        'null_count': int(df[mapped_col].isnull().sum())
                    }
                    stats[col_name] = col_stats
                except:
                    continue

        return {
            "mapped_columns": mapping,
            "numeric_validation": {
                "invalid_columns": invalid_columns,
                "valid_columns": [col for col in mapping.keys() if col not in invalid_columns]
            },
            "data_stats": stats,
            "message": "Validation completed successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating file: {str(e)}"
        )

@app.post("/process/{filename}")
async def process_file(filename: str, mapping: Dict[str, str], token: str = Depends(verify_token)):
    """
    Process a file using the provided column mapping.
    Standardizes the data format and prepares it for fraud detection.
    Also adds synthetic fraud labels for training purposes.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Read and process the file
        df = pd.read_csv(file_path)

        # Rename columns according to mapping
        df_processed = df.rename(columns={v: k for k, v in mapping.items()})

        # Handle missing required columns
        required_columns = ['TransactionAmount', 'TransactionDuration',
                          'LoginAttempts', 'AccountBalance', 'CustomerAge']

        # Calculate global statistics for reference
        global_stats = {}
        for col in required_columns:
            if col in df_processed.columns:
                global_stats[col] = {
                    'mean': float(df_processed[col].mean()) if not df_processed[col].empty else 0,
                    'std': float(df_processed[col].std()) if not df_processed[col].empty else 1
                }

        # Fill missing columns with defaults
        for col in required_columns:
            if col not in df_processed.columns:
                if col == 'TransactionDuration':
                    df_processed[col] = 60  # Default 60 seconds
                    global_stats[col] = {'mean': 60, 'std': 1}
                elif col == 'LoginAttempts':
                    df_processed[col] = 1   # Default 1 attempt
                    global_stats[col] = {'mean': 1, 'std': 1}
                elif col == 'CustomerAge':
                    df_processed[col] = 35  # Default age
                    global_stats[col] = {'mean': 35, 'std': 1}
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required column: {col}"
                    )

        # Normalize numeric features
        numeric_columns = {
            'TransactionAmount': 'amount_normalized',
            'TransactionDuration': 'duration_normalized',
            'LoginAttempts': 'attempts_normalized',
            'AccountBalance': 'balance_normalized',
            'CustomerAge': 'age_normalized'
        }

        for original_col, normalized_col in numeric_columns.items():
            if original_col in df_processed.columns:
                # Apply simple z-score normalization
                mean_val = global_stats[original_col]['mean']
                std_val = global_stats[original_col]['std']
                if std_val == 0:  # Handle constant values
                    std_val = 1
                df_processed[normalized_col] = (df_processed[original_col] - mean_val) / std_val

                # Clip values to valid range (-5, 5)
                df_processed[normalized_col] = df_processed[normalized_col].clip(-5, 5)

        # Generate synthetic fraud labels based on known fraud patterns
        def generate_fraud_label(row, stats):
            # High-risk patterns
            is_fraud = False

            amount_mean = stats['TransactionAmount']['mean']

            # Pattern 1: High amount transaction
            if row['TransactionAmount'] > amount_mean * 2:
                is_fraud = True

            # Pattern 2: Amount much larger than account balance
            elif 'AccountBalance' in row and row['TransactionAmount'] > row['AccountBalance'] * 0.8:
                is_fraud = True

            # Pattern 3: Unusual login attempts for elderly customers
            elif ('LoginAttempts' in row and 'CustomerAge' in row and
                  row['LoginAttempts'] > 3 and row['CustomerAge'] > 60):
                is_fraud = True

            # Pattern 4: Very quick large transactions
            elif ('TransactionDuration' in row and
                  row['TransactionAmount'] > amount_mean * 1.5 and
                  row['TransactionDuration'] < stats['TransactionDuration']['mean'] * 0.3):
                is_fraud = True

            # Add random noise to prevent overfitting
            if np.random.random() < 0.02:  # 2% random flip
                is_fraud = not is_fraud

            return float(is_fraud)

        # Apply fraud labeling
        df_processed['FraudLabel'] = df_processed.apply(lambda row: generate_fraud_label(row, global_stats), axis=1)

        # Save processed file
        processed_filename = f"processed_{filename}"
        processed_path = UPLOAD_DIR / processed_filename
        df_processed.to_csv(processed_path, index=False)

        # Calculate fraud statistics
        total_transactions = len(df_processed)
        fraud_count = df_processed['FraudLabel'].sum()
        fraud_percentage = (fraud_count / total_transactions) * 100

        return {
            "status": "success",
            "processed_filename": processed_filename,
            "rows_processed": len(df_processed),
            "columns_standardized": list(df_processed.columns),
            "fraud_statistics": {
                "total_transactions": total_transactions,
                "fraud_transactions": int(fraud_count),
                "fraud_percentage": float(fraud_percentage)
            },
            "global_stats": global_stats
        }

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/files/")
async def list_files(token: str = Depends(verify_token)):
    """List all uploaded and processed files"""
    try:
        files = []
        for file_path in UPLOAD_DIR.glob("*.csv"):
            stats = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size_bytes": stats.st_size,
                "uploaded_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "is_processed": file_path.name.startswith("processed_")
            })
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/status")
async def get_training_status(token: str = Depends(verify_token)):
    """Get the current status of model training."""
    global training_status

    try:
        # Ensure training_status exists and has all required fields
        if not isinstance(training_status, dict):
            training_status = {
                "is_training": False,
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 50,
                "current_loss": 0,
                "best_loss": float('inf'),
                "error": None,
                "message": "Not started",
                "start_time": None,
                "end_time": None
            }

        # Return current status
        return {
            "status": "success",
            "training_status": {
                "is_training": training_status.get("is_training", False),
                "progress": training_status.get("progress", 0),
                "current_epoch": training_status.get("current_epoch", 0),
                "total_epochs": training_status.get("total_epochs", 50),
                "current_loss": training_status.get("current_loss", 0),
                "best_loss": training_status.get("best_loss", float('inf')),
                "error": training_status.get("error"),
                "message": training_status.get("message", "Not started"),
                "start_time": training_status.get("start_time"),
                "end_time": training_status.get("end_time")
            }
        }

    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        # Return a default status in case of error
        return {
            "status": "error",
            "training_status": {
                "is_training": False,
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 50,
                "current_loss": 0,
                "best_loss": float('inf'),
                "error": str(e),
                "message": "Error getting training status",
                "start_time": None,
                "end_time": None
            }
        }

async def train_model_task(filename: str):
    """Background task for model training."""
    global training_status
    try:
        # Reset training status
        training_status.update({
            "is_training": True,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 50,
            "current_loss": 0,
            "best_loss": float('inf'),
            "error": None,
            "message": "Training started",
            "start_time": datetime.now().isoformat(),
            "end_time": None
        })

        # Load processed data
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise ValueError(f"File {filename} not found")

        df = pd.read_csv(file_path)

        # Validate FraudLabel exists
        if 'FraudLabel' not in df.columns:
            raise ValueError("FraudLabel column not found in processed data")

        # Get normalized features
        feature_cols = [col for col in df.columns if col.endswith('_normalized')]
        if not feature_cols:
            raise ValueError("No normalized features found")

        X = df[feature_cols].values
        y = df['FraudLabel'].values

        # Training configuration
        epochs = training_status["total_epochs"]
        patience = 5
        patience_counter = 0
        best_loss = float('inf')

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        try:
            # Initialize model and optimizer
            model = model_manager.model
            optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            # Training loop
            for epoch in range(epochs):
                if not training_status["is_training"]:
                    logger.info("Training stopped by user")
                    training_status["message"] = "Training stopped by user"
                    break

                training_status["current_epoch"] = epoch + 1
                model.model.train()

                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                training_status["current_loss"] = current_loss

                # Early stopping check
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    model_manager.save_model()
                    training_status["best_loss"] = best_loss
                    training_status["message"] = f"New best loss: {best_loss:.4f}"
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    training_status["message"] = f"Early stopping at epoch {epoch + 1}"
                    break

                # Update progress
                training_status["progress"] = ((epoch + 1) / epochs) * 100
                training_status["message"] = f"Training epoch {epoch + 1}/{epochs}, loss: {current_loss:.4f}"

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.1)

            # Final update
            training_status.update({
                "is_training": False,
                "progress": 100,
                "message": "Training completed successfully",
                "end_time": datetime.now().isoformat()
            })
            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Error during training loop: {str(e)}")
            training_status.update({
                "is_training": False,
                "error": f"Training error: {str(e)}",
                "message": f"Training failed: {str(e)}",
                "end_time": datetime.now().isoformat()
            })
            raise

    except Exception as e:
        logger.error(f"Error in training task: {str(e)}")
        training_status.update({
            "is_training": False,
            "error": str(e),
            "message": f"Training failed: {str(e)}",
            "end_time": datetime.now().isoformat()
        })

@app.post("/train/{filename}")
async def train_model(
    filename: str,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Train the fraud detection model using processed data.
    Training runs in the background and progress can be monitored via /train/status endpoint.
    """
    try:
        # Check if training is already in progress
        if training_status["is_training"]:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "detail": "Training already in progress",
                    "current_progress": training_status["progress"]
                }
            )

        # Validate file exists before starting training
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File {filename} not found"
            )

        # Start training in background
        background_tasks.add_task(train_model_task, filename)

        return JSONResponse(
            status_code=202,
            content={
                "status": "success",
                "message": "Training started successfully",
                "monitor_url": "/train/status"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting training: {str(e)}"
        )

@app.post("/predict/{filename}")
async def predict_fraud(
    filename: str,
    threshold: float = 0.5,
    max_details: int = 100,
    token: str = Depends(verify_token)
):
    """
    Make fraud predictions on a processed file.
    Returns summary information and details of top suspicious transactions.
    The threshold may be automatically adjusted if the fraud rate is unreasonably high.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Read processed data
        df = pd.read_csv(file_path)

        # Validate and extract features
        try:
            X = df[model_manager.expected_features].values
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature: {str(e)}"
            )

        # Make predictions
        try:
            prediction_result = model_manager.predict(X, threshold)
            predictions = prediction_result["predictions"]
            probabilities = prediction_result["probabilities"]
            adjusted_threshold = prediction_result.get("adjusted_threshold", threshold)

            # Add warning if threshold was adjusted
            threshold_warning = None
            if adjusted_threshold != threshold:
                threshold_warning = f"Threshold was automatically adjusted from {threshold} to {adjusted_threshold} to maintain reasonable fraud detection rate"
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )

        # Add predictions to dataframe
        df['fraud_probability'] = probabilities
        df['is_fraud'] = predictions

        # Get indices of fraud predictions sorted by probability
        fraud_indices = np.where(predictions == 1)[0]
        fraud_probs = probabilities[fraud_indices]

        # Sort by probability (highest first) and limit to max_details
        top_fraud_indices = fraud_indices[np.argsort(-fraud_probs)][:max_details]

        # Calculate probability distribution statistics
        prob_stats = {
            "mean": float(np.mean(probabilities)),
            "median": float(np.median(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
            "percentiles": {
                "25": float(np.percentile(probabilities, 25)),
                "75": float(np.percentile(probabilities, 75)),
                "90": float(np.percentile(probabilities, 90)),
                "95": float(np.percentile(probabilities, 95)),
                "99": float(np.percentile(probabilities, 99))
            }
        }

        # Prepare detailed fraud information
        fraud_details = []
        for idx in top_fraud_indices:
            transaction_data = {
                "TransactionAmount": float(df.iloc[idx].get("TransactionAmount", 0)),
                "TransactionDuration": float(df.iloc[idx].get("TransactionDuration", 0)),
                "LoginAttempts": float(df.iloc[idx].get("LoginAttempts", 0)),
                "AccountBalance": float(df.iloc[idx].get("AccountBalance", 0)),
                "CustomerAge": float(df.iloc[idx].get("CustomerAge", 0))
            }

            fraud_details.append({
                "index": int(idx),
                "probability": float(probabilities[idx]),
                "confidence_score": float(abs(probabilities[idx] - 0.5) * 2),  # Convert to 0-1 scale
                "normalized_features": {
                    name: float(X[idx][i])
                    for i, name in enumerate(model_manager.expected_features)
                },
                "transaction_data": transaction_data
            })

        # Save predictions
        output_filename = f"predictions_{filename}"
        output_path = UPLOAD_DIR / output_filename
        df.to_csv(output_path, index=False)

        response = {
            "status": "success",
            "output_filename": output_filename,
            "predictions_summary": {
                "total_transactions": len(df),
                "fraud_detected": int(np.sum(predictions)),
                "legitimate_transactions": int(len(df) - np.sum(predictions)),
                "fraud_percentage": float(np.sum(predictions) / len(df) * 100),
                "probability_stats": prob_stats,
                "confidence_stats": {
                    "avg_fraud_confidence": float(np.mean(probabilities[predictions == 1])) if np.any(predictions == 1) else 0,
                    "avg_legitimate_confidence": float(1 - np.mean(probabilities[predictions == 0])) if np.any(predictions == 0) else 0
                },
                "threshold": {
                    "requested": threshold,
                    "actual": adjusted_threshold,
                    "warning": threshold_warning
                }
            },
            "fraud_transactions": {
                "total_count": int(np.sum(predictions)),
                "details_shown": len(fraud_details),
                "details": fraud_details
            }
        }

        # Add recommendations if fraud rate is still high
        fraud_rate = np.sum(predictions) / len(df)
        if fraud_rate > 0.10:  # More than 10% fraud rate
            response["recommendations"] = [
                "The current fraud detection rate is unusually high. Consider:",
                f"1. Increasing the threshold (current: {adjusted_threshold})",
                "2. Reviewing the model's training data for potential bias",
                "3. Implementing additional validation rules",
                "4. Collecting more legitimate transaction data for training"
            ]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )

@app.get("/evaluate/{filename}")
async def evaluate_model(filename: str, token: str = Depends(verify_token)):
    """
    Evaluate model performance on a processed file.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Read processed data
        df = pd.read_csv(file_path)

        # Extract features and actual fraud labels
        feature_columns = [col for col in df.columns if col.endswith('_normalized') or col.endswith('_encoded')]
        X = df[feature_columns].values
        y = df['FraudScore'].values >= df['FraudScore'].quantile(0.95)  # Top 5% as fraud

        # Evaluate model
        metrics = model_manager.evaluate(X, y)

        return {
            "status": "success",
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/status")
async def get_model_status(token: str = Depends(verify_token)):
    """
    Get the current status of the model.
    """
    try:
        return {
            "status": "success",
            "model_info": {
                "input_dimensions": model_manager.input_dim,
                "device": str(model_manager.device),
                "model_file_exists": model_manager.model_path.exists(),
                "scaler_file_exists": model_manager.scaler_path.exists()
            }
        }
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}