"""
Custom XGBoost model for Jane Street market prediction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

from .base_model import BaseModel
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    Custom XGBoost model with hyperparameter tuning for financial time series.
    Optimized for Jane Street market prediction competition.
    """
    
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: int = 1,
                 gamma: float = 0,
                 scale_pos_weight: Optional[float] = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 early_stopping_rounds: int = 20,
                 validation_size: float = 0.2):
        """
        Initialize XGBoost model with financial-specific parameters.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required for split
            scale_pos_weight: Balance positive/negative weights (auto if None)
            random_state: Random seed
            n_jobs: Number of parallel jobs
            early_stopping_rounds: Early stopping rounds
            validation_size: Validation split size
        """
        super().__init__("XGBoostModel")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size
        
        # Model and training artifacts
        self.model = None
        self.feature_importances = None
        self.training_history = None
        self.best_iteration = None
        self.validation_score = None
        
    def _prepare_params(self, pos_weight: Optional[float] = None) -> Dict[str, Any]:
        """
        Prepare XGBoost parameters.
        
        Args:
            pos_weight: Positive class weight
            
        Returns:
            Parameter dictionary
        """
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': 0,  # Reduce output noise
            'tree_method': 'hist'  # Fast histogram-based algorithm
        }
        
        # Set scale_pos_weight if provided
        if pos_weight is not None:
            params['scale_pos_weight'] = pos_weight
            
        return params
    
    def _calculate_pos_weight(self, y: pd.Series) -> float:
        """
        Calculate positive class weight for imbalanced dataset.
        
        Args:
            y: Target series
            
        Returns:
            Positive class weight
        """
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            return 1.0
            
        pos_weight = neg_count / pos_count
        logger.info(f"Class balance - Negative: {neg_count}, Positive: {pos_count}")
        logger.info(f"Calculated pos_weight: {pos_weight:.3f}")
        
        return pos_weight
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """
        Train XGBoost model with early stopping.
        
        Args:
            X: Feature dataframe
            y: Target series (binary)
            
        Returns:
            Self for method chaining
        """
        self.validate_input(X)
        
        if not isinstance(y, pd.Series):
            raise TypeError("Target must be pandas Series")
            
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Target must be binary (0/1)")
            
        logger.info(f"Training XGBoost on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Calculate class weights if needed
        pos_weight = self.scale_pos_weight
        if pos_weight is None:
            pos_weight = self._calculate_pos_weight(y)
            
        # Prepare parameters
        params = self._prepare_params(pos_weight)
        
        # Train/validation split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        # Evaluation list
        evallist = [(dtrain, 'train'), (dval, 'val')]
        
        # Training with early stopping
        logger.info(f"Training with early stopping (patience={self.early_stopping_rounds})")
        
        evals_result = {}
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evallist,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        # Store training artifacts
        self.best_iteration = self.model.best_iteration
        self.training_history = evals_result
        self.validation_score = self.model.best_score
        
        # Get feature importance
        self.feature_importances = self.model.get_score(importance_type='weight')
        
        self.is_fitted = True
        
        # Log training results
        logger.info(f"Training complete:")
        logger.info(f"  - Best iteration: {self.best_iteration}")
        logger.info(f"  - Best validation AUC: {self.validation_score:.4f}")
        logger.info(f"  - Feature importance computed for {len(self.feature_importances)} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (binary decisions).
        
        Args:
            X: Feature dataframe
            
        Returns:
            Binary predictions (0/1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Probability array [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.validate_input(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)
        
        # Get probability predictions
        probabilities_pos = self.model.predict(
            dtest, 
            iteration_range=(0, self.best_iteration + 1)
        )
        
        # Return as [prob_class_0, prob_class_1] format
        probabilities = np.column_stack([1 - probabilities_pos, probabilities_pos])
        
        return probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # Normalize importance scores
        total_importance = sum(self.feature_importances.values())
        if total_importance == 0:
            return {feature: 0.0 for feature in self.feature_names}
            
        normalized_importance = {
            feature: importance / total_importance 
            for feature, importance in self.feature_importances.items()
        }
        
        # Fill missing features with 0
        for feature in self.feature_names:
            if feature not in normalized_importance:
                normalized_importance[feature] = 0.0
                
        return normalized_importance
    
    def get_feature_importance_dataframe(self) -> pd.DataFrame:
        """
        Get feature importance as sorted DataFrame.
        
        Returns:
            DataFrame with features and importance scores
        """
        importance_dict = self.get_feature_importance()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history.
        
        Returns:
            Dictionary with training and validation metrics over iterations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting training history")
            
        return self.training_history
    
    def plot_importance(self, max_features: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            max_features: Maximum number of features to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance_dataframe()
            top_features = importance_df.head(max_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {max_features} Feature Importance (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        
        Returns:
            Dictionary with model parameters
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save XGBoost model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'XGBoostModel':
        """
        Load XGBoost model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self for method chaining
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
