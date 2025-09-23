"""Configuration management using Pydantic for type safety."""

from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data loading and processing configuration."""
    train_path: str
    feature_prefix: str = "feature_"
    target_column: str = "resp"
    weight_column: str = "weight"
    date_column: str = "date"
    n_features: int = 130


class PCAConfig(BaseModel):
    """PCA configuration for dimensionality reduction."""
    variance_threshold: float = 0.95
    random_state: int = 42


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    pca: PCAConfig


class AutoencoderConfig(BaseModel):
    """Autoencoder model configuration."""
    encoding_dim: int = 64
    hidden_dims: List[int] = [128, 64, 32]
    dropout_rate: float = 0.2
    epochs: int = 50
    batch_size: int = 1024


class XGBoostConfig(BaseModel):
    """XGBoost model configuration."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    random_state: int = 42


class EnsembleConfig(BaseModel):
    """Ensemble model configuration."""
    autoencoder_weight: float = 0.3
    xgboost_weight: float = 0.7


class ModelConfig(BaseModel):
    """Model configuration container."""
    autoencoder: AutoencoderConfig
    xgboost: XGBoostConfig
    ensemble: EnsembleConfig


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


class Config(BaseModel):
    """Main configuration class."""
    data: DataConfig
    features: FeatureConfig
    models: ModelConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()


# Global config instance
_config = None

def get_config(config_path: str = "config/model_config.yaml") -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_yaml(config_path)
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
