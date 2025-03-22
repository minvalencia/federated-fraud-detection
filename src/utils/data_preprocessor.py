import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.required_columns = {
            'TransactionAmount',
            'TransactionDuration',
            'LoginAttempts',
            'AccountBalance',
            'CustomerAge'
        }
        self.categorical_columns = [
            'TransactionType',
            'Channel',
            'CustomerOccupation'
        ]

    def load_and_preprocess_bank_data(self, data_path: str, bank_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess data from a specific bank.

        Args:
            data_path: Path to the bank's data file
            bank_id: Identifier for the bank

        Returns:
            Preprocessed DataFrame and preprocessing statistics
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path} with {len(df)} rows")

            # Store original column names for reference
            original_columns = df.columns.tolist()

            # Check and standardize required columns
            missing_columns = self.required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Add bank identifier
            df['BankID'] = bank_id

            # Handle missing values
            df = self._handle_missing_values(df)

            # Feature engineering
            df = self._engineer_features(df)

            # Normalize numerical features
            df, scaling_params = self._normalize_features(df)

            # Encode categorical variables if present
            df, encoding_mapping = self._encode_categorical_features(df)

            # Calculate fraud score based on multiple indicators
            df['FraudScore'] = self._calculate_fraud_score(df)

            preprocessing_stats = {
                'original_columns': original_columns,
                'scaling_params': scaling_params,
                'encoding_mapping': encoding_mapping,
                'row_count': len(df),
                'bank_id': bank_id
            }

            return df, preprocessing_stats

        except Exception as e:
            logger.error(f"Error processing data for bank {bank_id}: {str(e)}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Numerical columns: fill with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())

        # Categorical columns: fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data."""
        # Transaction amount percentile
        df['AmountPercentile'] = df['TransactionAmount'].rank(pct=True)

        # Transaction velocity (if timestamp available)
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            df['HourOfDay'] = df['TransactionDate'].dt.hour
            df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek

        # Account balance ratio
        df['BalanceRatio'] = df['TransactionAmount'] / df['AccountBalance']

        # Login risk score
        df['LoginRiskScore'] = np.where(df['LoginAttempts'] > 2,
                                      df['LoginAttempts'] * 2,
                                      df['LoginAttempts'])

        return df

    def _normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize numerical features using robust scaling."""
        numerical_features = [
            'TransactionAmount',
            'TransactionDuration',
            'AccountBalance',
            'CustomerAge'
        ]

        scaling_params = {}
        for feature in numerical_features:
            if feature in df.columns:
                median = df[feature].median()
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1

                # Store scaling parameters
                scaling_params[feature] = {
                    'median': median,
                    'iqr': iqr
                }

                # Apply robust scaling
                df[f'{feature}_normalized'] = (df[feature] - median) / iqr

        return df, scaling_params

    def _encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical features using label encoding."""
        encoding_mapping = {}

        for col in self.categorical_columns:
            if col in df.columns:
                # Create mapping for unique values
                unique_values = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}

                # Store mapping
                encoding_mapping[col] = mapping

                # Apply encoding
                df[f'{col}_encoded'] = df[col].map(mapping)

        return df, encoding_mapping

    def _calculate_fraud_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fraud risk score based on multiple indicators."""
        score = pd.Series(0, index=df.index)

        # Amount-based risk
        score += 2.0 * (df['AmountPercentile'] > 0.9).astype(float)

        # Login attempt risk
        score += 1.5 * (df['LoginAttempts'] > 2).astype(float)

        # Transaction duration risk
        score += 1.0 * (df['TransactionDuration'] > df['TransactionDuration'].quantile(0.9)).astype(float)

        # Balance ratio risk
        score += 1.5 * (df['BalanceRatio'] > df['BalanceRatio'].quantile(0.9)).astype(float)

        return score

    @staticmethod
    def combine_bank_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine preprocessed data from multiple banks."""
        return pd.concat(dataframes, ignore_index=True)

def process_all_banks(data_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process all bank data files in the specified directory.

    Args:
        data_dir: Directory containing bank data files

    Returns:
        Combined preprocessed DataFrame and preprocessing statistics
    """
    preprocessor = DataPreprocessor()
    processed_dfs = []
    all_stats = {}

    # Process each data file
    for file_path in Path(data_dir).glob('*.csv'):
        bank_id = file_path.stem
        logger.info(f"Processing data for bank: {bank_id}")

        try:
            df, stats = preprocessor.load_and_preprocess_bank_data(str(file_path), bank_id)
            processed_dfs.append(df)
            all_stats[bank_id] = stats
            logger.info(f"Successfully processed {len(df)} rows for bank {bank_id}")

        except Exception as e:
            logger.error(f"Failed to process {bank_id}: {str(e)}")
            continue

    if not processed_dfs:
        raise ValueError("No data was successfully processed")

    # Combine all processed data
    combined_df = DataPreprocessor.combine_bank_data(processed_dfs)
    logger.info(f"Combined dataset has {len(combined_df)} rows")

    return combined_df, all_stats