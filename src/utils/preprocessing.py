# Import required libraries for data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path: str):
    """
    Load and clean the bank transaction dataset.
    Handles missing values and prepares data for training.

    Args:
        file_path (str): Path to the CSV dataset file

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset
    """
    # Load raw data from CSV file
    df = pd.read_csv(file_path)

    # Handle missing values by filling with mean values
    # This preserves the statistical properties of the features
    df = df.fillna(df.mean())

    # Note: Add additional preprocessing steps based on specific dataset characteristics
    # For example:
    # - Handle categorical variables
    # - Remove outliers
    # - Feature engineering
    # - Data normalization

    return df

def prepare_data_for_training(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for model training by splitting into train/test sets
    and scaling features.

    Args:
        df (pd.DataFrame): Input DataFrame with transaction data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Separate features and target variable
    X = df.drop(['fraud_label'], axis=1)
    y = df['fraud_label']

    # Split data into training and testing sets
    # Using stratification to maintain fraud/non-fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features using StandardScaler
    # This ensures all features are on the same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def split_data_for_clients(X: np.ndarray, y: np.ndarray, num_clients: int):
    """
    Split data into balanced portions for different clients (banks).
    Ensures each client gets a representative sample of fraud cases.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        num_clients (int): Number of clients to split data between

    Returns:
        list: List of (X, y) tuples for each client
    """
    # Separate fraud and non-fraud indices
    # This allows us to maintain class balance across clients
    fraud_indices = np.where(y == 1)[0]
    non_fraud_indices = np.where(y == 0)[0]

    # Shuffle indices to ensure random distribution
    np.random.shuffle(fraud_indices)
    np.random.shuffle(non_fraud_indices)

    # Calculate number of samples per client
    fraud_per_client = len(fraud_indices) // num_clients
    non_fraud_per_client = len(non_fraud_indices) // num_clients

    # Distribute data to clients
    client_data = []
    for i in range(num_clients):
        # Get balanced subset of fraud and non-fraud cases for this client
        client_fraud_idx = fraud_indices[i*fraud_per_client:(i+1)*fraud_per_client]
        client_non_fraud_idx = non_fraud_indices[i*non_fraud_per_client:(i+1)*non_fraud_per_client]
        client_indices = np.concatenate([client_fraud_idx, client_non_fraud_idx])

        # Shuffle the client's data
        np.random.shuffle(client_indices)

        # Extract this client's portion of the dataset
        client_X = X[client_indices]
        client_y = y[client_indices]

        client_data.append((client_X, client_y))

    return client_data