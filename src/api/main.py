from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
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
from .ml_model import model_manager

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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Upload and process a bank transaction data file.
    Automatically maps columns and validates data.
    """
    try:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bank_data_{timestamp}.csv"
        file_path = UPLOAD_DIR / filename

        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the file
        df = pd.read_csv(file_path)

        # Guess column mapping
        mapping = ColumnMapper.guess_column_mapping(df)

        # Validate numeric columns
        invalid_columns = ColumnMapper.validate_numeric_columns(df, mapping)

        # Calculate mapping coverage
        required_columns = set(['TransactionAmount', 'TransactionDuration',
                              'LoginAttempts', 'AccountBalance', 'CustomerAge'])
        mapped_required = set(mapping.keys()) & required_columns
        coverage = len(mapped_required) / len(required_columns)

        return {
            "status": "success",
            "filename": filename,
            "total_rows": len(df),
            "column_mapping": mapping,
            "mapping_coverage": coverage,
            "invalid_columns": invalid_columns,
            "missing_required_columns": list(required_columns - mapped_required)
        }

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/validate/{filename}")
async def validate_mapping(filename: str, mapping: Dict[str, str], token: str = Depends(verify_token)):
    """
    Validate a proposed column mapping for a specific file.
    Returns detailed statistics about the data quality.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        df = pd.read_csv(file_path)

        # Validate numeric columns
        invalid_columns = ColumnMapper.validate_numeric_columns(df, mapping)

        # Calculate basic statistics for numeric columns
        stats = {}
        for standard_col, file_col in mapping.items():
            if standard_col in ['TransactionAmount', 'TransactionDuration',
                              'LoginAttempts', 'AccountBalance', 'CustomerAge']:
                try:
                    series = pd.to_numeric(df[file_col])
                    stats[standard_col] = {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "null_count": int(series.isnull().sum())
                    }
                except:
                    stats[standard_col] = {"error": "Invalid numeric data"}

        return {
            "status": "success",
            "invalid_columns": invalid_columns,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error validating mapping: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process/{filename}")
async def process_file(filename: str, mapping: Dict[str, str], token: str = Depends(verify_token)):
    """
    Process a file using the provided column mapping.
    Standardizes the data format and prepares it for fraud detection.
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

        for col in required_columns:
            if col not in df_processed.columns:
                # Use median values from historical data or reasonable defaults
                if col == 'TransactionDuration':
                    df_processed[col] = 60  # Default 60 seconds
                elif col == 'LoginAttempts':
                    df_processed[col] = 1   # Default 1 attempt
                elif col == 'CustomerAge':
                    df_processed[col] = 35  # Default age
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required column: {col}"
                    )

        # Save processed file
        processed_filename = f"processed_{filename}"
        processed_path = UPLOAD_DIR / processed_filename
        df_processed.to_csv(processed_path, index=False)

        return {
            "status": "success",
            "processed_filename": processed_filename,
            "rows_processed": len(df_processed),
            "columns_standardized": list(df_processed.columns)
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

@app.post("/train/{filename}")
async def train_model(filename: str, token: str = Depends(verify_token)):
    """
    Train the model using a processed file.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Read processed data
        df = pd.read_csv(file_path)

        # Extract features and target
        feature_columns = [col for col in df.columns if col.endswith('_normalized') or col.endswith('_encoded')]
        X = df[feature_columns].values
        y = df['FraudScore'].values >= df['FraudScore'].quantile(0.95)  # Top 5% as fraud

        # Train model
        training_stats = model_manager.train(X, y)

        return {
            "status": "success",
            "message": "Model trained successfully",
            "training_stats": training_stats
        }

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{filename}")
async def predict_fraud(filename: str, threshold: float = 0.5, token: str = Depends(verify_token)):
    """
    Make fraud predictions on a processed file.
    Returns detailed information about detected fraudulent transactions.
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Read processed data
        df = pd.read_csv(file_path)

        # Extract features and validate column names
        feature_columns = [col for col in df.columns if col.endswith('_normalized') or col.endswith('_encoded')]

        # Validate that we have the correct features
        if len(feature_columns) != model_manager.input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_manager.input_dim} features, got {len(feature_columns)}"
            )

        # Extract features in the correct order
        try:
            X = df[model_manager.expected_features].values
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature: {str(e)}"
            )

        # Make predictions with validation
        try:
            prediction_result = model_manager.predict(X, threshold, feature_names=model_manager.expected_features)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Add predictions to dataframe
        df['fraud_probability'] = prediction_result['probabilities']
        df['is_fraud'] = prediction_result['predictions']

        # Enhance fraud details with original transaction data
        enhanced_fraud_details = []
        for fraud in prediction_result['fraud_details']:
            idx = fraud['index']
            transaction_data = df.iloc[idx].to_dict()

            # Remove normalized features from transaction data to avoid duplication
            for feat in model_manager.expected_features:
                transaction_data.pop(feat, None)

            enhanced_fraud_details.append({
                "index": idx,
                "probability": fraud['probability'],
                "normalized_features": fraud['features'],
                "transaction_data": transaction_data
            })

        # Save predictions
        output_filename = f"predictions_{filename}"
        output_path = UPLOAD_DIR / output_filename
        df.to_csv(output_path, index=False)

        # Return detailed summary
        return {
            "status": "success",
            "output_filename": output_filename,
            "predictions_summary": prediction_result['summary'],
            "validation_status": "All validations passed",
            "fraud_transactions": {
                "count": len(enhanced_fraud_details),
                "details": enhanced_fraud_details
            }
        }

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

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