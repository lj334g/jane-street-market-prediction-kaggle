"""Abstract base model following SOLID principles."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all models following SOLID principles."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = None
        self.model = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Train the model on input features and target.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Self for method chaining
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Predictions array
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Probability predictions array
        """
        pass
        
    def validate_input(self, X: pd.DataFrame) -> None:
        """Validate input features."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if self.is_fitted and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
                
        if X.empty:
            raise ValueError("Input DataFrame is empty")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        return None
        
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']
        self.is_fitted = model_data['is_fitted']
        
        return self
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config.copy()
        
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        return self
        
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')"


class BaseEnsemble(BaseModel):
    """Abstract base class for ensemble models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.base_models = {}
        self.weights = {}
        
    @abstractmethod
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> 'BaseEnsemble':
        """
        Add a base model to the ensemble.
        
        Args:
            name: Model identifier
            model: Base model instance
            weight: Model weight in ensemble
            
        Returns:
            Self for method chaining
        """
        pass
        
    @abstractmethod
    def combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions from base models.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            
        Returns:
            Combined predictions
        """
        pass
        
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights."""
        return self.weights.copy()
        
    def set_model_weights(self, weights: Dict[str, float]) -> 'BaseEnsemble':
        """
        Set model weights.
        
        Args:
            weights: Dictionary mapping model names to weights
            
        Returns:
            Self for method chaining
        """
        # Validate weights
        missing_models = set(weights.keys()) - set(self.base_models.keys())
        if missing_models:
            raise ValueError(f"Unknown models in weights: {missing_models}")
            
        self.weights.update(weights)
        return self
