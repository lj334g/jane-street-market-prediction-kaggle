"""PCA dimensionality reduction with inference speed benchmarking."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import time
import logging

from ..utils.timer import InferenceSpeedBenchmark

logger = logging.getLogger(__name__)


class PCADimensionalityReducer:
    def __init__(self, 
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 random_state: int = 42,
                 scale_features: bool = True):
        """
        Initialize PCA reducer.
        
        Args:
            n_components: Number of components (if None, use variance_threshold)
            variance_threshold: Minimum variance to retain
            random_state: Random seed for reproducibility
            scale_features: Whether to standardize features before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.scale_features = scale_features
        
        self.pca = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names_in = None
        self.feature_names_out = None
        self.is_fitted = False
        
        # Performance metrics
        self.reduction_metrics = {}
        
    def fit(self, X: pd.DataFrame) -> 'PCADimensionalityReducer':
        """
        Fit PCA on training data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
            
        self.feature_names_in = X.columns.tolist()
        
        # Scale features if requested
        if self.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()
            
        # Initialize PCA
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        else:
            # Use variance threshold to determine components
            temp_pca = PCA(random_state=self.random_state)
            temp_pca.fit(X_scaled)
            
            # Find number of components for desired variance
            cumsum_var = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
            
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Create output feature names
        self.feature_names_out = [f'pca_{i}' for i in range(self.pca.n_components_)]
        
        # Calculate reduction metrics
        self._calculate_reduction_metrics(X.shape[1])
        
        self.is_fitted = True
        logger.info(f"PCA fitted: {X.shape[1]} -> {self.pca.n_components_} features "
                   f"({self.reduction_metrics['reduction_percentage']:.1f}% reduction)")
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed dataframe with reduced dimensions
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transforming")
            
        # Validate input features
        if not all(col in X.columns for col in self.feature_names_in):
            missing = set(self.feature_names_in) - set(X.columns)
            raise ValueError(f"Missing features: {missing}")
            
        # Select and order features consistently
        X_ordered = X[self.feature_names_in]
        
        # Scale if needed
        if self.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_ordered),
                columns=self.feature_names_in,
                index=X.index
            )
        else:
            X_scaled = X_ordered
            
        # Apply PCA transformation
        X_transformed = self.pca.transform(X_scaled)
        
        # Return as DataFrame
        return pd.DataFrame(
            X_transformed,
            columns=self.feature_names_out,
            index=X.index
        )
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.fit(X).transform(X)
        
    def _calculate_reduction_metrics(self, original_dims: int) -> None:
        """Calculate dimensionality reduction metrics."""
        reduced_dims = self.pca.n_components_
        
        self.reduction_metrics = {
            'original_dimensions': original_dims,
            'reduced_dimensions': reduced_dims,
            'reduction_percentage': (original_dims - reduced_dims) / original_dims * 100,
            'variance_explained': self.pca.explained_variance_ratio_.sum(),
            'components_for_95_variance': np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.95) + 1
        }
        
    def benchmark_inference_speed(self, 
                                 n_samples: int = 1000, 
                                 n_trials: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed improvement from dimensionality reduction.
        
        Args:
            n_samples: Number of samples for benchmarking
            n_trials: Number of trials for averaging
            
        Returns:
            Dictionary with speed benchmark results
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before benchmarking")
            
        original_dims = self.reduction_metrics['original_dimensions']
        reduced_dims = self.reduction_metrics['reduced_dimensions']
        
        benchmark = InferenceSpeedBenchmark(n_samples=n_samples, n_trials=n_trials)
        speed_results = benchmark.benchmark_inference_speed(original_dims, reduced_dims)
        
        # Combine with reduction metrics
        combined_results = {
            **self.reduction_metrics,
            **speed_results,
            'meets_4x_speedup': speed_results['speedup_factor'] >= 4.0,
            'meets_80_reduction': self.reduction_metrics['reduction_percentage'] >= 80.0
        }
        
        return combined_results
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from PCA loadings.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting feature importance")
            
        # Calculate feature importance as sum of absolute loadings weighted by variance explained
        loadings = self.pca.components_.T  # features x components
        variance_weights = self.pca.explained_variance_ratio_
        
        # Weighted importance
        importance_scores = np.sum(np.abs(loadings) * variance_weights, axis=1)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names_in,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    def get_component_analysis(self) -> pd.DataFrame:
        """
        Analyze PCA components and their variance contribution.
        
        Returns:
            DataFrame with component analysis
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before component analysis")
            
        component_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(self.pca.n_components_)],
            'variance_explained': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'eigenvalue': self.pca.explained_variance_
        })
        
        return component_df
        
    def inverse_transform(self, X_reduced: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform reduced data back to original space.
        
        Args:
            X_reduced: Reduced feature dataframe
            
        Returns:
            Reconstructed dataframe in original feature space
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse transform")
            
        # Apply inverse PCA transformation
        X_reconstructed = self.pca.inverse_transform(X_reduced)
        
        # Inverse scale if needed
        if self.scale_features:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
            
        return pd.DataFrame(
            X_reconstructed,
            columns=self.feature_names_in,
            index=X_reduced.index
        )
        
    def get_reconstruction_error(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate reconstruction error for each sample.
        
        Args:
            X: Original feature dataframe
            
        Returns:
            Series with reconstruction error per sample
        """
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        
        # Calculate MSE per sample
        mse_per_sample = ((X[self.feature_names_in] - X_reconstructed) ** 2).mean(axis=1)
        
        return mse_per_sample
