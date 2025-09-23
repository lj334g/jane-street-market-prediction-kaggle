"""
Ensemble model blending Autoencoder-MLP and XGBoost.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

from .base_model import BaseEnsemble
from .autoencoder_mlp import AutoencoderMLP
from .xgboost_model import XGBoostModel
from ..evaluation.utility_metric import JaneStreetUtilityMetric

logger = logging.getLogger(__name__)


class AutoencoderXGBoostEnsemble(BaseEnsemble):
    """
    Weighted ensemble of Autoencoder-MLP and XGBoost models.
    Optimized for Jane Street market prediction utility metric.
    """
    
    def __init__(self,
                 autoencoder_weight: float = 0.3,
                 xgboost_weight: float = 0.7,
                 autoencoder_config: Optional[Dict[str, Any]] = None,
                 xgboost_config: Optional[Dict[str, Any]] = None,
                 optimize_weights: bool = True,
                 weight_optimization_cv: int = 3,
                 random_state: int = 42):
        """
        Initialize ensemble model.
        
        Args:
            autoencoder_weight: Weight for autoencoder predictions
            xgboost_weight: Weight for XGBoost predictions
            autoencoder_config: Configuration for autoencoder model
            xgboost_config: Configuration for XGBoost model
            optimize_weights: Whether to optimize ensemble weights
            weight_optimization_cv: CV folds for weight optimization
            random_state: Random seed
        """
        super().__init__("AutoencoderXGBoostEnsemble")
        
        # Validate weights
        if not np.isclose(autoencoder_weight + xgboost_weight, 1.0):
            logger.warning(f"Weights don't sum to 1.0: {autoencoder_weight + xgboost_weight}")
            # Normalize weights
            total_weight = autoencoder_weight + xgboost_weight
            autoencoder_weight /= total_weight
            xgboost_weight /= total_weight
            logger.info(f"Normalized weights: AE={autoencoder_weight:.3f}, XGB={xgboost_weight:.3f}")
        
        self.autoencoder_weight = autoencoder_weight
        self.xgboost_weight = xgboost_weight
        self.optimize_weights = optimize_weights
        self.weight_optimization_cv = weight_optimization_cv
        self.random_state = random_state
        
        # Initialize models
        ae_config = autoencoder_config or {}
        xgb_config = xgboost_config or {}
        
        self.autoencoder = AutoencoderMLP(random_state=random_state, **ae_config)
        self.xgboost = XGBoostModel(random_state=random_state, **xgb_config)
        
        # Add models to base ensemble
        self.add_model('autoencoder', self.autoencoder, autoencoder_weight)
        self.add_model('xgboost', self.xgboost, xgboost_weight)
        
        # Training artifacts
        self.optimized_weights = None
        self.cv_scores = None
        self.utility_improvement = None
        
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> 'AutoencoderXGBoostEnsemble':
        """Add model to ensemble."""
        self.base_models[name] = model
        self.weights[name] = weight
        return self
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoencoderXGBoostEnsemble':
        """
        Train both models and optionally optimize ensemble weights.
        
        Args:
            X: Feature dataframe
            y: Target series (binary)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names for validation
        self.feature_names = X.columns.tolist()
        
        # Train individual models
        logger.info("Training Autoencoder-MLP...")
        self.autoencoder.fit(X, y)
        
        logger.info("Training XGBoost...")
        self.xgboost.fit(X, y)
        
        # Optimize weights if requested
        if self.optimize_weights:
            logger.info("Optimizing ensemble weights...")
            self._optimize_ensemble_weights(X, y)
        
        self.is_fitted = True
        
        logger.info("Ensemble training complete")
        logger.info(f"Final weights - AE: {self.weights['autoencoder']:.3f}, "
                   f"XGB: {self.weights['xgboost']:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Probability array [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        self.validate_input(X)
        
        # Get individual model predictions
        ae_proba = self.autoencoder.predict_proba(X)
        xgb_proba = self.xgboost.predict_proba(X)
        
        # Weighted combination
        ensemble_proba = (self.weights['autoencoder'] * ae_proba + 
                         self.weights['xgboost'] * xgb_proba)
        
        return ensemble_proba
    
    def combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions from base models using weights.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            
        Returns:
            Combined predictions
        """
        if not set(predictions.keys()).issubset(set(self.weights.keys())):
            missing = set(predictions.keys()) - set(self.weights.keys())
            raise ValueError(f"Unknown models in predictions: {missing}")
        
        # Initialize combined predictions
        combined = np.zeros_like(next(iter(predictions.values())))
        total_weight = 0
        
        # Weighted sum
        for model_name, pred in predictions.items():
            weight = self.weights[model_name]
            combined += weight * pred
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if not np.isclose(total_weight, 1.0):
            combined /= total_weight
            
        return combined
    
    def _optimize_ensemble_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Optimize ensemble weights using cross-validation.
        Uses AUC as optimization metric.
        """
        from scipy.optimize import minimize
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.weight_optimization_cv, shuffle=True, 
                           random_state=self.random_state)
        
        def objective(weights):
            """Objective function to minimize (negative AUC)."""
            w_ae, w_xgb = weights[0], 1.0 - weights[0]  # Constrain to sum to 1
            
            cv_scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train models on fold
                ae_fold = AutoencoderMLP(random_state=self.random_state)
                xgb_fold = XGBoostModel(random_state=self.random_state)
                
                ae_fold.fit(X_train, y_train)
                xgb_fold.fit(X_train, y_train)
                
                # Get predictions
                ae_proba = ae_fold.predict_proba(X_val)[:, 1]
                xgb_proba = xgb_fold.predict_proba(X_val)[:, 1]
                
                # Weighted ensemble
                ensemble_proba = w_ae * ae_proba + w_xgb * xgb_proba
                
                # Calculate AUC
                auc = roc_auc_score(y_val, ensemble_proba)
                cv_scores.append(auc)
            
            return -np.mean(cv_scores)  # Minimize negative AUC
        
        # Optimize weights (constrain autoencoder weight between 0 and 1)
        result = minimize(
            objective,
            x0=[self.autoencoder_weight],
            bounds=[(0.0, 1.0)],
            method='L-BFGS-B'
        )
        
        if result.success:
            optimized_ae_weight = result.x[0]
            optimized_xgb_weight = 1.0 - optimized_ae_weight
            
            self.optimized_weights = {
                'autoencoder': optimized_ae_weight,
                'xgboost': optimized_xgb_weight
            }
            
            # Update weights
            self.weights['autoencoder'] = optimized_ae_weight
            self.weights['xgboost'] = optimized_xgb_weight
            
            optimal_auc = -result.fun
            logger.info(f"Weight optimization successful:")
            logger.info(f"  - Optimized AE weight: {optimized_ae_weight:.3f}")
            logger.info(f"  - Optimized XGB weight: {optimized_xgb_weight:.3f}")
            logger.info(f"  - Cross-validation AUC: {optimal_auc:.4f}")
            
        else:
            logger.warning("Weight optimization failed, using initial weights")
            self.optimized_weights = self.get_model_weights()
    
    def evaluate_utility_improvement(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   weights: pd.Series,
                                   returns: pd.Series,
                                   baseline_model: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate utility improvement over baseline.
        
        Args:
            X: Feature dataframe
            y: Target series
            weights: Sample weights
            returns: Sample returns
            baseline_model: Baseline model (uses XGBoost alone if None)
            
        Returns:
            Dictionary with utility comparison results
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")
        
        # Use XGBoost alone as baseline if not provided
        if baseline_model is None:
            baseline_model = self.xgboost
        
        # Calculate utilities
        utility_metric = JaneStreetUtilityMetric()
        
        ensemble_preds = self.predict_proba(X)[:, 1]
        baseline_preds = baseline_model.predict_proba(X)[:, 1]
        
        ensemble_utility = utility_metric.calculate_utility(
            y.values, ensemble_preds, weights.values, returns.values
        )
        baseline_utility = utility_metric.calculate_utility(
            y.values, baseline_preds, weights.values, returns.values
        )
        
        improvement_pct = utility_metric.calculate_utility_improvement(
            baseline_utility, ensemble_utility
        )
        
        results = {
            'ensemble_utility': ensemble_utility,
            'baseline_utility': baseline_utility,
            'absolute_improvement': ensemble_utility - baseline_utility,
            'improvement_percentage': improvement_pct,
            'meets_12_percent_target': improvement_pct >= 12.0
        }
        
        self.utility_improvement = results
        
        logger.info(f"Utility evaluation results:")
        logger.info(f"  - Ensemble utility: {ensemble_utility:.6f}")
        logger.info(f"  - Baseline utility: {baseline_utility:.6f}")
        logger.info(f"  - Improvement: {improvement_pct:.1f}%")
        logger.info(f"  - Meets 12% target: {results['meets_12_percent_target']}")
        
        return results
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual models for analysis.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary with individual model predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting individual predictions")
        
        return {
            'autoencoder': self.autoencoder.predict_proba(X)[:, 1],
            'xgboost': self.xgboost.predict_proba(X)[:, 1],
            'ensemble': self.predict_proba(X)[:, 1]
        }
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze individual model contributions to ensemble prediction.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with model contributions
        """
        predictions = self.get_individual_predictions(X)
        
        contributions_df = pd.DataFrame({
            'autoencoder_pred': predictions['autoencoder'],
            'xgboost_pred': predictions['xgboost'],
            'autoencoder_contribution': predictions['autoencoder'] * self.weights['autoencoder'],
            'xgboost_contribution': predictions['xgboost'] * self.weights['xgboost'],
            'ensemble_pred': predictions['ensemble'],
            'autoencoder_weight': self.weights['autoencoder'],
            'xgboost_weight': self.weights['xgboost']
        })
        
        return contributions_df
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training results and metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting training summary")
        
        summary = {
            'model_type': 'AutoencoderXGBoostEnsemble',
            'n_base_models': len(self.base_models),
            'base_models': list(self.base_models.keys()),
            'initial_weights': {
                'autoencoder': self.autoencoder_weight,
                'xgboost': self.xgboost_weight
            },
            'final_weights': self.get_model_weights(),
            'weight_optimization_applied': self.optimize_weights,
            'optimized_weights': self.optimized_weights
        }
        
        # Add utility improvement if available
        if self.utility_improvement is not None:
            summary['utility_improvement'] = self.utility_improvement
        
        return summary
