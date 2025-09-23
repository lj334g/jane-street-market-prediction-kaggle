"""
Technical indicators for creating predictive signals from HFT data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine:
    """
    Create technical indicators and predictive signals from raw features.
    Designed for Jane Street market prediction competition data.
    """
    
    def __init__(self, 
                 feature_prefix: str = "feature_",
                 lookback_windows: List[int] = [5, 10, 20],
                 quantile_windows: List[int] = [50, 100],
                 scale_features: bool = True):
        """
        Initialize technical indicator engine.
        
        Args:
            feature_prefix: Prefix for feature columns
            lookback_windows: Windows for rolling statistics
            quantile_windows: Windows for quantile-based features
            scale_features: Whether to scale generated features
        """
        self.feature_prefix = feature_prefix
        self.lookback_windows = lookback_windows
        self.quantile_windows = quantile_windows
        self.scale_features = scale_features
        
        self.feature_columns = None
        self.scaler = RobustScaler() if scale_features else None
        self.is_fitted = False
        
    def _identify_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify feature columns from dataframe."""
        feature_cols = [col for col in data.columns if col.startswith(self.feature_prefix)]
        logger.info(f"Identified {len(feature_cols)} feature columns")
        return feature_cols
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            data: Input dataframe with features
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        lag_features = pd.DataFrame(index=data.index)
        
        for col in self.feature_columns:
            for lag in lags:
                lag_features[f'{col}_lag_{lag}'] = data[col].shift(lag)
                
        logger.info(f"Created {lag_features.shape[1]} lagged features")
        return lag_features
    
    def create_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            DataFrame with rolling statistical features
        """
        rolling_features = pd.DataFrame(index=data.index)
        
        for col in self.feature_columns:
            series = data[col]
            
            for window in self.lookback_windows:
                # Rolling mean
                rolling_features[f'{col}_mean_{window}'] = series.rolling(window).mean()
                
                # Rolling standard deviation
                rolling_features[f'{col}_std_{window}'] = series.rolling(window).std()
                
                # Rolling min/max
                rolling_features[f'{col}_min_{window}'] = series.rolling(window).min()
                rolling_features[f'{col}_max_{window}'] = series.rolling(window).max()
                
                # Z-score (current value vs rolling mean)
                rolling_mean = series.rolling(window).mean()
                rolling_std = series.rolling(window).std()
                rolling_features[f'{col}_zscore_{window}'] = (series - rolling_mean) / rolling_std
                
                # Percentage change from rolling mean
                rolling_features[f'{col}_pct_from_mean_{window}'] = (series / rolling_mean - 1)
                
        logger.info(f"Created {rolling_features.shape[1]} rolling statistical features")
        return rolling_features
    
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum and trend features.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            DataFrame with momentum features
        """
        momentum_features = pd.DataFrame(index=data.index)
        
        for col in self.feature_columns:
            series = data[col]
            
            # Simple returns
            for window in [1, 2, 5, 10]:
                momentum_features[f'{col}_return_{window}'] = series.pct_change(window)
                
            # Momentum (price relative to past)
            for window in self.lookback_windows:
                momentum_features[f'{col}_momentum_{window}'] = series / series.shift(window) - 1
                
            # Rate of change
            for window in [3, 7, 14]:
                momentum_features[f'{col}_roc_{window}'] = (series - series.shift(window)) / series.shift(window)
                
            # Trend strength (linear regression slope)
            for window in [10, 20]:
                def rolling_slope(x):
                    if len(x) < 2 or x.isna().all():
                        return np.nan
                    y = np.arange(len(x))
                    valid_mask = ~x.isna()
                    if valid_mask.sum() < 2:
                        return np.nan
                    slope, _, _, _, _ = stats.linregress(y[valid_mask], x[valid_mask])
                    return slope
                
                momentum_features[f'{col}_trend_{window}'] = series.rolling(window).apply(rolling_slope)
                
        logger.info(f"Created {momentum_features.shape[1]} momentum features")
        return momentum_features
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility and dispersion features.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            DataFrame with volatility features
        """
        volatility_features = pd.DataFrame(index=data.index)
        
        for col in self.feature_columns:
            series = data[col]
            returns = series.pct_change()
            
            for window in self.lookback_windows:
                # Rolling volatility (standard deviation of returns)
                volatility_features[f'{col}_volatility_{window}'] = returns.rolling(window).std()
                
                # Average True Range (ATR) proxy
                high = series.rolling(window).max()
                low = series.rolling(window).min()
                volatility_features[f'{col}_atr_{window}'] = (high - low) / series.rolling(window).mean()
                
                # Coefficient of variation
                rolling_mean = series.rolling(window).mean()
                rolling_std = series.rolling(window).std()
                volatility_features[f'{col}_cv_{window}'] = rolling_std / rolling_mean
                
        logger.info(f"Created {volatility_features.shape[1]} volatility features")
        return volatility_features
    
    def create_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-sectional features comparing across all features.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            DataFrame with cross-sectional features
        """
        cross_features = pd.DataFrame(index=data.index)
        feature_data = data[self.feature_columns]
        
        # Cross-sectional rankings
        cross_features['rank_mean'] = feature_data.rank(axis=1, pct=True).mean(axis=1)
        cross_features['rank_std'] = feature_data.rank(axis=1, pct=True).std(axis=1)
        
        # Cross-sectional statistics
        cross_features['feature_mean'] = feature_data.mean(axis=1)
        cross_features['feature_std'] = feature_data.std(axis=1)
        cross_features['feature_median'] = feature_data.median(axis=1)
        cross_features['feature_skew'] = feature_data.skew(axis=1)
        cross_features['feature_kurt'] = feature_data.kurtosis(axis=1)
        
        # Number of features above/below certain thresholds
        cross_features['n_positive'] = (feature_data > 0).sum(axis=1)
        cross_features['n_negative'] = (feature_data < 0).sum(axis=1)
        cross_features['n_extreme_positive'] = (feature_data > feature_data.quantile(0.95, axis=1).values[:, None]).sum(axis=1)
        cross_features['n_extreme_negative'] = (feature_data < feature_data.quantile(0.05, axis=1).values[:, None]).sum(axis=1)
        
        logger.info(f"Created {cross_features.shape[1]} cross-sectional features")
        return cross_features
    
    def create_quantile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create quantile-based features.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            DataFrame with quantile features
        """
        quantile_features = pd.DataFrame(index=data.index)
        
        for col in self.feature_columns:
            series = data[col]
            
            for window in self.quantile_windows:
                # Rolling quantiles
                quantile_features[f'{col}_q25_{window}'] = series.rolling(window).quantile(0.25)
                quantile_features[f'{col}_q75_{window}'] = series.rolling(window).quantile(0.75)
                quantile_features[f'{col}_iqr_{window}'] = (
                    series.rolling(window).quantile(0.75) - series.rolling(window).quantile(0.25)
                )
                
                # Position relative to quantiles
                q25 = series.rolling(window).quantile(0.25)
                q75 = series.rolling(window).quantile(0.75)
                quantile_features[f'{col}_above_q75_{window}'] = (series > q75).astype(int)
                quantile_features[f'{col}_below_q25_{window}'] = (series < q25).astype(int)
                
        logger.info(f"Created {quantile_features.shape[1]} quantile features")
        return quantile_features
    
    def create_interaction_features(self, data: pd.DataFrame, max_interactions: int = 50) -> pd.DataFrame:
        """
        Create interaction features between top features.
        
        Args:
            data: Input dataframe with features
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        interaction_features = pd.DataFrame(index=data.index)
        
        # Use a subset of features for interactions to avoid explosion
        n_features = min(10, len(self.feature_columns))
        selected_features = self.feature_columns[:n_features]
        
        interaction_count = 0
        for i, feat1 in enumerate(selected_features):
            if interaction_count >= max_interactions:
                break
                
            for j, feat2 in enumerate(selected_features[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                # Multiplication
                interaction_features[f'{feat1}_{feat2}_mult'] = data[feat1] * data[feat2]
                interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
                
                # Division (with protection against zero)
                denominator = data[feat2].replace(0, np.nan)
                interaction_features[f'{feat1}_{feat2}_div'] = data[feat1] / denominator
                interaction_count += 1
                
        logger.info(f"Created {interaction_features.shape[1]} interaction features")
        return interaction_features
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators and predictive signals.
        
        Args:
            data: Input dataframe with raw features
            
        Returns:
            DataFrame with all generated features
        """
        logger.info("Creating technical indicators and predictive signals...")
        
        # Identify feature columns
        self.feature_columns = self._identify_feature_columns(data)
        
        if len(self.feature_columns) == 0:
            raise ValueError("No feature columns found with specified prefix")
        
        # Create all feature types
        all_features = [data[self.feature_columns]]  # Start with original features
        
        # Lag features
        lag_feats = self.create_lag_features(data)
        all_features.append(lag_feats)
        
        # Rolling statistics
        rolling_feats = self.create_rolling_statistics(data)
        all_features.append(rolling_feats)
        
        # Momentum features
        momentum_feats = self.create_momentum_features(data)
        all_features.append(momentum_feats)
        
        # Volatility features
        volatility_feats = self.create_volatility_features(data)
        all_features.append(volatility_feats)
        
        # Cross-sectional features
        cross_feats = self.create_cross_sectional_features(data)
        all_features.append(cross_feats)
        
        # Quantile features
        quantile_feats = self.create_quantile_features(data)
        all_features.append(quantile_feats)
        
        # Interaction features
        interaction_feats = self.create_interaction_features(data)
        all_features.append(interaction_feats)
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=1)
        
        # Handle infinite values
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
        
        # Scale features if requested
        if self.scale_features:
            logger.info("Scaling features...")
            
            # Fit scaler on non-NaN values
            feature_array = combined_features.values
            finite_mask = np.isfinite(feature_array)
            
            if finite_mask.any():
                # Create a copy for fitting (only finite values)
                fit_data = feature_array.copy()
                fit_data[~finite_mask] = 0  # Temporarily fill for fitting
                
                self.scaler.fit(fit_data)
                
                # Transform all data
                scaled_data = self.scaler.transform(feature_array)
                
                # Restore NaN where they were originally
                scaled_data[~finite_mask] = np.nan
                
                combined_features = pd.DataFrame(
                    scaled_data, 
                    index=combined_features.index,
                    columns=combined_features.columns
                )
        
        self.is_fitted = True
        
        logger.info(f"Technical indicators created:")
        logger.info(f"  - Original features: {len(self.feature_columns)}")
        logger.info(f"  - Total features: {combined_features.shape[1]}")
        logger.info(f"  - Feature expansion: {combined_features.shape[1] / len(self.feature_columns):.1f}x")
        
        return combined_features
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted parameters.
        
        Args:
            data: New dataframe to transform
            
        Returns:
            DataFrame with generated features
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
            
        return self.fit_transform(data)  # For now, recompute (could be optimized)
    
    def get_feature_summary(self) -> Dict[str, int]:
        """
        Get summary of created feature types.
        
        Returns:
            Dictionary with feature counts by type
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
            
        return {
            'original_features': len(self.feature_columns),
            'lag_features': len(self.feature_columns) * 3,  # 3 default lags
            'rolling_features': len(self.feature_columns) * len(self.lookback_windows) * 6,  # 6 stats per window
            'momentum_features': len(self.feature_columns) * (4 + len(self.lookback_windows) * 3),
            'volatility_features': len(self.feature_columns) * len(self.lookback_windows) * 3,
            'cross_sectional_features': 9,  # Fixed number
            'quantile_features': len(self.feature_columns) * len(self.quantile_windows) * 5,
            'interaction_features': 50  # Default max
        }
