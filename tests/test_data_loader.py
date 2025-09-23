"""Unit tests for HFT data loader."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.loader import HFTDataLoader, load_jane_street_data
from src.utils.config import DataConfig


class TestHFTDataLoader:
    """Test cases for HFT data loader."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample data configuration for testing."""
        return DataConfig(
            train_path="test_data.csv",
            feature_prefix="feature_",
            target_column="resp",
            weight_column="weight",
            date_column="date",
            n_features=5  # Smaller for testing
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample Jane Street-like data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'date': np.random.randint(0, 100, n_samples),
            'weight': np.random.uniform(0.5, 2.0, n_samples),
            'resp': np.random.normal(0, 0.1, n_samples),
            'ts_id': range(n_samples)
        })
        
        # Add feature columns
        for i in range(5):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
            
        # Add some missing values and zero weights for testing
        data.iloc[0:10, data.columns.get_loc('feature_0')] = np.nan
        data.iloc[0:5, data.columns.get_loc('weight')] = 0.0
        
        return data
    
    @pytest.fixture
    def csv_file(self, sample_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
            
        # Cleanup
        Path(f.name).unlink()
    
    @pytest.fixture
    def zip_file(self, csv_file):
        """Create temporary ZIP file with CSV data."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with zipfile.ZipFile(f.name, 'w') as zf:
                zf.write(csv_file, 'train.csv')
            yield f.name
            
        # Cleanup
        Path(f.name).unlink()
        
    def test_init(self, sample_config):
        """Test loader initialization."""
        loader = HFTDataLoader("test_path.csv", sample_config.dict())
        
        assert loader.data_path == Path("test_path.csv")
        assert loader.config.feature_prefix == "feature_"
        assert loader.feature_columns is None
        
    def test_load_raw_data_csv(self, csv_file, sample_config):
        """Test loading data from CSV file."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        data = loader.load_raw_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert 'resp' in data.columns
        assert 'weight' in data.columns
        
    def test_load_raw_data_zip(self, zip_file, sample_config):
        """Test loading data from ZIP file."""
        loader = HFTDataLoader(zip_file, sample_config.dict())
        data = loader.load_raw_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        
    def test_validate_data_structure(self, csv_file, sample_config):
        """Test data structure validation."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        data = loader.load_raw_data()
        
        validation_results = loader.validate_data_structure(data)
        
        assert validation_results['total_rows'] == 1000
        assert validation_results['feature_count'] == 5
        assert validation_results['features_match'] is True
        assert validation_results['has_target'] is True
        assert validation_results['has_weight'] is True
        assert validation_results['trading_days'] is not None
        
    def test_preprocess_data(self, csv_file, sample_config):
        """Test data preprocessing."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        data = loader.load_raw_data()
        
        processed_data = loader.preprocess_data(data)
        
        # Check zero-weight samples were filtered
        assert len(processed_data) < len(data)
        assert all(processed_data['weight'] > 0)
        
        # Check missing values were filled
        assert processed_data['feature_0'].isnull().sum() == 0
        
    def test_create_target_variable(self, csv_file, sample_config):
        """Test target variable creation."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        data = loader.load_raw_data()
        processed_data = loader.preprocess_data(data)
        
        target = loader.create_target_variable(processed_data)
        
        assert target.dtype == int
        assert all(target.isin([0, 1]))
        assert len(target) == len(processed_data)
        
    def test_get_features_and_target(self, csv_file, sample_config):
        """Test feature and target extraction."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        data = loader.load_raw_data()
        processed_data = loader.preprocess_data(data)
        
        X, y = loader.get_features_and_target(processed_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[1] == 5  # 5 features
        assert len(X) == len(y)
        assert all(col.startswith('feature_') for col in X.columns)
        
    def test_load_processed_data_complete(self, csv_file, sample_config):
        """Test complete data loading pipeline."""
        loader = HFTDataLoader(csv_file, sample_config.dict())
        X, y, metadata = loader.load_processed_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(metadata, dict)
        
        assert X.shape[1] == 5  # Feature count
        assert len(X) == len(y)
        assert 'validation_results' in metadata
        assert 'feature_columns' in metadata
        
    def test_unsupported_file_format(self, sample_config):
        """Test error handling for unsupported file formats."""
        loader = HFTDataLoader("test_file.txt", sample_config.dict())
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_raw_data()
            
    def test_missing_target_column(self, csv_file, sample_config):
        """Test error handling for missing target column."""
        sample_config.target_column = "missing_column"
        loader = HFTDataLoader(csv_file, sample_config.dict())
        
        data = loader.load_raw_data()
        processed_data = loader.preprocess_data(data)
        
        with pytest.raises(ValueError, match="Target column"):
            loader.create_target_variable(processed_data)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def csv_file(self):
        """Create temporary CSV file."""
        data = pd.DataFrame({
            'date': [0, 1, 2],
            'weight': [1.0, 1.5, 0.8],
            'resp': [0.01, -0.005, 0.02],
            'feature_0': [1.0, 2.0, 3.0],
            'feature_1': [0.5, 1.5, 2.5]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            yield f.name
            
        Path(f.name).unlink()
        
    def test_load_jane_street_data(self, csv_file):
        """Test convenience function for loading data."""
        # Mock the config to have only 2 features for this test
        with patch('src.data.loader.get_config') as mock_config:
            mock_config.return_value.data.n_features = 2
            mock_config.return_value.data.feature_prefix = "feature_"
            mock_config.return_value.data.target_column = "resp"
            mock_config.return_value.data.weight_column = "weight"
            mock_config.return_value.data.date_column = "date"
            
            X, y, metadata = load_jane_street_data(csv_file)
            
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert isinstance(metadata, dict)
            assert X.shape[1] == 2  # 2 features


if __name__ == "__main__":
    pytest.main([__file__])
