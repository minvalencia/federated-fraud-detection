import pytest
from fastapi.testclient import TestClient
from ..api.main import app
import numpy as np
import pandas as pd
from pathlib import Path
import json

client = TestClient(app)

@pytest.fixture
def sample_transaction_data():
    return pd.DataFrame({
        'TransactionAmount': [1000.0, 5000.0, 200.0],
        'TransactionDuration': [60, 120, 30],
        'LoginAttempts': [1, 5, 1],
        'AccountBalance': [5000.0, 1000.0, 10000.0],
        'CustomerAge': [35, 40, 25]
    })

def test_upload_file(sample_transaction_data):
    # Create a temporary CSV file
    temp_file = Path("test_data.csv")
    sample_transaction_data.to_csv(temp_file, index=False)

    with open(temp_file, "rb") as f:
        response = client.post(
            "/upload/",
            files={"file": ("test_data.csv", f, "text/csv")},
            headers={"X-API-Key": "test-key"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "filename" in data
    assert data["total_rows"] == 3

def test_process_file():
    response = client.post(
        "/process/test_data.csv",
        json={
            "mapping": {
                "TransactionAmount": "amount",
                "TransactionDuration": "duration",
                "LoginAttempts": "attempts",
                "AccountBalance": "balance",
                "CustomerAge": "age"
            }
        },
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "processed_filename" in data

def test_train_model():
    response = client.post(
        "/train/test_data.csv",
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "training_stats" in data
    assert "losses" in data["training_stats"]

def test_predict_fraud():
    response = client.post(
        "/predict/test_data.csv",
        params={"threshold": 0.5},
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predictions_summary" in data
    assert "fraud_transactions" in data

    # Verify fraud details structure
    if data["fraud_transactions"]["count"] > 0:
        fraud_detail = data["fraud_transactions"]["details"][0]
        assert "index" in fraud_detail
        assert "probability" in fraud_detail
        assert "normalized_features" in fraud_detail
        assert "transaction_data" in fraud_detail

def test_evaluate_model():
    response = client.get(
        "/evaluate/test_data.csv",
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert all(metric in data["metrics"] for metric in [
        "accuracy", "precision", "recall", "f1_score"
    ])

def test_model_status():
    response = client.get(
        "/model/status",
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_info" in data
    assert all(key in data["model_info"] for key in [
        "input_dimensions", "device", "model_file_exists"
    ])

def test_invalid_file():
    response = client.post(
        "/predict/nonexistent.csv",
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 404

def test_invalid_features():
    # Create data with wrong features
    invalid_data = pd.DataFrame({
        'WrongFeature': [100.0],
        'AnotherWrong': [200.0]
    })

    temp_file = Path("invalid_data.csv")
    invalid_data.to_csv(temp_file, index=False)

    with open(temp_file, "rb") as f:
        response = client.post(
            "/upload/",
            files={"file": ("invalid_data.csv", f, "text/csv")},
            headers={"X-API-Key": "test-key"}
        )

    assert response.status_code == 400

def test_continuous_learning():
    # Test 1: Initial training
    train_response = client.post(
        "/train/test_data.csv",
        headers={"X-API-Key": "test-key"}
    )
    initial_metrics = client.get(
        "/evaluate/test_data.csv",
        headers={"X-API-Key": "test-key"}
    ).json()["metrics"]

    # Test 2: Add new data and retrain
    new_data = pd.DataFrame({
        'TransactionAmount': [10000.0],  # Suspicious amount
        'TransactionDuration': [300],    # Long duration
        'LoginAttempts': [10],           # Many attempts
        'AccountBalance': [100.0],       # Low balance
        'CustomerAge': [30]
    })

    temp_file = Path("new_data.csv")
    new_data.to_csv(temp_file, index=False)

    with open(temp_file, "rb") as f:
        upload_response = client.post(
            "/upload/",
            files={"file": ("new_data.csv", f, "text/csv")},
            headers={"X-API-Key": "test-key"}
        )

    # Process and train on new data
    process_response = client.post(
        "/process/new_data.csv",
        json={"mapping": {
            "TransactionAmount": "amount",
            "TransactionDuration": "duration",
            "LoginAttempts": "attempts",
            "AccountBalance": "balance",
            "CustomerAge": "age"
        }},
        headers={"X-API-Key": "test-key"}
    )

    train_response = client.post(
        "/train/new_data.csv",
        headers={"X-API-Key": "test-key"}
    )

    # Get updated metrics
    updated_metrics = client.get(
        "/evaluate/new_data.csv",
        headers={"X-API-Key": "test-key"}
    ).json()["metrics"]

    # Verify that the model has adapted
    assert train_response.status_code == 200
    assert "metrics" in updated_metrics

    # Test prediction on new suspicious transaction
    predict_response = client.post(
        "/predict/new_data.csv",
        headers={"X-API-Key": "test-key"}
    )

    prediction_data = predict_response.json()
    assert prediction_data["status"] == "success"
    assert prediction_data["fraud_transactions"]["count"] > 0  # Should detect the suspicious transaction