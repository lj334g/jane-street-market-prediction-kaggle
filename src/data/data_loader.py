"""High-frequency trading data loader for Jane Street competition data."""

import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from ..utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFTDataLoader:
    """
    Load and validate 500 days of high-frequency trading data across 130 features.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data file (CSV or ZIP)
            config: Optional configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config or get_config().data
        self.feature_columns = None
        self.data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw Jane Street data from CSV or ZIP file.
        
        Returns:
            Raw dataframe with all columns
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.zip':
                data = self._load_from_zip()
            elif self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
                
            logger.info(f"Loaded data shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_from_zip(self) -> pd.DataFrame:
        """Load data from ZIP file (as in original notebooks)."""
        with zipfile.ZipFile(self.data_path) as zf:
            # Look for train.csv in the zip
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV files found in ZIP archive")
                
            # Use the first CSV file or train.csv if it exists
            csv_file = 'train.csv' if 'train.csv' in csv_files else csv_files[0]
            logger.info(f"Reading {csv_file} from ZIP archive")
            
            with zf.open(csv_file) as csv_file_obj:
                data = pd.read_csv(csv_file_obj)
                
        return data
    
    def validate_data_structure(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data structure matches Jane Street competition format.
        
        Args:
            data: Raw dataframe
            
        Returns:
            Dictionary with validation results
        """
        # Find feature columns
        feature_cols = [col for col in data.columns if col.startswith(self.config.feature_prefix)]
        self.feature_columns = feature_cols
        
        # Validation checks
        validations = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'feature_count': len(feature_cols),
            'expected_features': self.config.n_features,
            'features_match': len(feature_cols) == self.config.n_features,
            'has_target': self.config.target_column in data.columns,
            'has_weight': self.config.weight_column in data.columns,
            'has_date': self.config.date_column in data.columns,
            'date_range': None,
            'trading_days': None
        }
        
        if validations['has_date']:
            validations['date_range'] = (data[self.config.date_column].min(), 
                                       data[self.config.date_column].max())
            validations['trading_days'] = data[self.config.date_column].nunique()
            
        # Log validation results
        logger.info(f"Data validation results:")
        logger.info(f"  - Total rows: {validations['total_rows']:,}")
        logger.info(f"  - Feature columns: {validations['feature_count']}/{validations['expected_features']}")
        logger.info(f"  - Trading days: {validations['trading_days']}")
        
        if not validations['features_match']:
            logger.warning(f"Expected {validations['expected_features']} features, "
                          f"found {validations['feature_count']}")
            
        return validations
    
    def preprocess_data(self, data: pd.DataFrame, 
                       memory_optimize: bool = True) -> pd.DataFrame:
        """
        Preprocess data for training (from original notebook logic).
        
        Args:
            data: Raw dataframe
            memory_optimize: Whether to optimize memory usage
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Preprocessing data...")
        
        # Memory optimization (from original notebook)
        if memory_optimize:
            float_cols = data.select_dtypes(include=['float64']).columns
            data = data.astype({col: np.float32 for col in float_cols})
            logger.info("Applied memory optimization (float64 -> float32)")
            
        # Filter out zero weight samples (from original logic)
        if self.config.weight_column in data.columns:
            original_rows = len(data)
            data = data[data[self.config.weight_column] != 0]
            filtered_rows = len(data)
            logger.info(f"Filtered zero-weight samples: {original_rows} -> {filtered_rows} rows")
            
        # Handle missing values (from original notebook)
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            logger.info(f"Found missing values in {len(missing_counts[missing_counts > 0])} columns")
            
            # Fill missing values with mean for features (original approach)
            feature_cols = [col for col in data.columns if col.startswith(self.config.feature_prefix)]
            for col in feature_cols:
                if data[col].isnull().any():
                    data[col].fillna(data[col].mean(), inplace=True)
                    
        return data
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable (from original notebook).
        
        Args:
            data: Preprocessed dataframe
            
        Returns:
            Binary target series (1 = profitable trade, 0 = no trade)
        """
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")
            
        # Binary classification: positive return = 1, else 0
        target = (data[self.config.target_column] > 0).astype(int)
        
        logger.info(f"Created binary target: {target.sum()} positive samples "
                   f"({target.mean():.1%} positive rate)")
        
        return target
    
    def get_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix and target vector.
        
        Args:
            data: Preprocessed dataframe
            
        Returns:
            Tuple of (features, target)
        """
        # Get feature columns
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns 
                                   if col.startswith(self.config.feature_prefix)]
            
        X = data[self.feature_columns].copy()
        y = self.create_target_variable(data)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Complete data loading and processing pipeline.
        
        Returns:
            Tuple of (features, target, metadata)
        """
        # Load and validate raw data
        raw_data = self.load_raw_data()
        validation_results = self.validate_data_structure(raw_data)
        
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Extract features and target
        X, y = self.get_features_and_target(processed_data)
        
        # Prepare metadata
        metadata = {
            'validation_results': validation_results,
            'feature_columns': self.feature_columns,
            'data_shape': processed_data.shape,
            'feature_shape': X.shape,
            'target_shape': y.shape,
            'positive_rate': y.mean(),
            'has_weights': self.config.weight_column in processed_data.columns
        }
        
        # Store processed data for potential reuse
        self.data = processed_data
        
        return X, y, metadata


def load_jane_street_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Convenience function to load Jane Street data.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Tuple of (features, target, metadata)
    """
    loader = HFTDataLoader(data_path)
    return loader.load_processed_data()
