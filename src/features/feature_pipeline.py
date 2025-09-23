"""
Complete feature processing pipeline combining technical indicators and PCA.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .technical_indicators import TechnicalIndicatorEngine
from .pca_reducer import PCADimensionalityReducer

logger = logging.getLogger(__name__)


class FeatureProcessingPipeline:
    """
    Complete feature processing pipeline for Jane Street data.
    
    Pipeline stages:
    1. Technical indicator creation
    2. Missing value imputation
    3. Feature selection/filtering
    4. Dimensionality reduction (PCA)
    5. Final scaling
    """
    
    def __init__(self,
                 create_technical_indicators: bool = True,
                 apply_pca: bool = True,
                 pca_variance_threshold: float = 0.95,
                 imputation_strategy: str = 'median',
                 final_scaling: bool = True,
                 remove_low_variance: bool = True,
                 variance_threshold: float = 0.01,
                 remove_highly_correlated: bool = True,
                 correlation_threshold: float = 0.95,
                 feature_prefix: str = "feature_",
                 random_state: int = 42):
        """
        Initialize feature processing pipeline.
        
        Args:
            create_technical_indicators: Whether to create technical indicators
            apply_pca: Whether to apply PCA dimensionality reduction
            pca_variance_threshold: PCA variance threshold
            imputation_strategy: Strategy for missing value imputation
            final_scaling: Whether to apply final scaling
            remove_low_variance: Whether to remove low variance features
            variance_threshold: Threshold for low variance feature removal
            remove_highly_correlated: Whether to remove highly correlated features
            correlation_threshold: Threshold for correlation-based feature removal
            feature_prefix: Prefix for feature columns
            random_state: Random seed
        """
        self.create_technical_indicators = create_technical_indicators
        self.apply_pca = apply_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.imputation_strategy = imputation_strategy
        self.final_scaling = final_scaling
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold
        self.remove_highly_correlated = remove_highly_correlated
        self.correlation_threshold = correlation_threshold
        self.feature_prefix = feature_prefix
        self.random_state = random_state
        
        # Pipeline components
        self.technical_engine = None
        self.imputer = None
        self.pca_reducer = None
        self.final_scaler = None
        
        # Feature selection artifacts
        self.low_variance_features = None
        self.high_correlation_features = None
        self.selected_features = None
        
        # Pipeline state
        self.is_fitted = False
        self.pipeline_steps = []
        self.feature_evolution = {}
        
    def _log_feature_evolution(self, step: str, features: pd.DataFrame) -> None:
        """Log feature evolution through pipeline."""
        self.feature_evolution[step] = {
            'n_features': features.shape[1],
            'n_samples': features.shape[0],
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': features.isnull().sum().sum(),
            'missing_percentage': features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100
        }
        
        logger.info(f"After {step}: {features.shape[1]} features, "
                   f"{self.feature_evolution[step]['missing_percentage']:.1f}% missing")
    
    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators."""
        if not self.create_technical_indicators:
            return data
            
        logger.info("Step 1: Creating technical indicators...")
        
        self.technical_engine = TechnicalIndicatorEngine(
            feature_prefix=self.feature_prefix,
            scale_features=False  # We'll scale later
        )
        
        enhanced_data = self.technical_engine.fit_transform(data)
        self._log_feature_evolution("technical_indicators", enhanced_data)
        self.pipeline_steps.append("technical_indicators")
        
        return enhanced_data
    
    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        missing_count = data.isnull().sum().sum()
        if missing_count == 0:
            logger.info("Step 2: No missing values to impute")
            return data
            
        logger.info(f"Step 2: Imputing {missing_count:,} missing values...")
        
        self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        
        # Fit and transform
        imputed_array = self.imputer.fit_transform(data)
        imputed_data = pd.DataFrame(
            imputed_array, 
            index=data.index, 
            columns=data.columns
        )
        
        self._log_feature_evolution("imputation", imputed_data)
        self.pipeline_steps.append("imputation")
        
        return imputed_data
    
    def _remove_low_variance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance."""
        if not self.remove_low_variance:
            return data
            
        logger.info("Step 3: Removing low variance features...")
        
        # Calculate variance for each feature
        feature_variances = data.var()
        low_variance_mask = feature_variances <= self.variance_threshold
        
        self.low_variance_features = feature_variances[low_variance_mask].index.tolist()
        
        if len(self.low_variance_features) > 0:
            logger.info(f"Removing {len(self.low_variance_features)} low variance features")
            data_filtered = data.drop(columns=self.low_variance_features)
        else:
            logger.info("No low variance features to remove")
            data_filtered = data
            
        self._log_feature_evolution("low_variance_removal", data_filtered)
        self.pipeline_steps.append("low_variance_removal")
        
        return data_filtered
    
    def _remove_highly_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        if not self.remove_highly_correlated:
            return data
            
        logger.info("Step 4: Removing highly correlated features...")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= self.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # Remove features with lower variance in highly correlated pairs
        self.high_correlation_features = []
        feature_variances = data.var()
        
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 not in self.high_correlation_features and feat2 not in self.high_correlation_features:
                # Keep the feature with higher variance
                if feature_variances[feat1] < feature_variances[feat2]:
                    self.high_correlation_features.append(feat1)
                else:
                    self.high_correlation_features.append(feat2)
        
        if len(self.high_correlation_features) > 0:
            logger.info(f"Removing {len(self.high_correlation_features)} highly correlated features")
            data_filtered = data.drop(columns=self.high_correlation_features)
        else:
            logger.info("No highly correlated features to remove")
            data_filtered = data
            
        self._log_feature_evolution("correlation_removal", data_filtered)
        self.pipeline_steps.append("correlation_removal")
        
        return data_filtered
    
    def _apply_pca_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA dimensionality reduction."""
        if not self.apply_pca:
            return data
            
        logger.info("Step 5: Applying PCA dimensionality reduction...")
        
        self.pca_reducer = PCADimensionalityReducer(
            variance_threshold=self.pca_variance_threshold,
            random_state=self.random_state,
            scale_features=True
        )
        
        reduced_data = self.pca_reducer.fit_transform(data)
        self._log_feature_evolution("pca_reduction", reduced_data)
        self.pipeline_steps.append("pca_reduction")
        
        return reduced_data
    
    def _apply_final_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply final feature scaling."""
        if not self.final_scaling or self.apply_pca:  # Skip if PCA already scaled
            return data
            
        logger.info("Step 6: Applying final feature scaling...")
        
        self.final_scaler = StandardScaler()
        scaled_array = self.final_scaler.fit_transform(data)
        
        scaled_data = pd.DataFrame(
            scaled_array,
            index=data.index,
            columns=data.columns
        )
        
        self._log_feature_evolution("final_scaling", scaled_data)
        self.pipeline_steps.append("final_scaling")
        
        return scaled_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete feature processing pipeline.
        
        Args:
            data: Raw input dataframe
            
        Returns:
            Processed feature dataframe
        """
        logger.info("Starting feature processing pipeline...")
        logger.info(f"Input data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Initialize tracking
        self.feature_evolution = {}
        self.pipeline_steps = []
        
        # Log initial state
        self._log_feature_evolution("input", data)
        
        # Execute pipeline steps
        processed_data = data.copy()
        
        # Step 1: Create technical indicators
        processed_data = self._create_technical_indicators(processed_data)
        
        # Step 2: Impute missing values
        processed_data = self._impute_missing_values(processed_data)
        
        # Step 3: Remove low variance features
        processed_data = self._remove_low_variance_features(processed_data)
        
        # Step 4: Remove highly correlated features
        processed_data = self._remove_highly_correlated_features(processed_data)
        
        # Step 5: Apply PCA
        processed_data = self._apply_pca_reduction(processed_data)
        
        # Step 6: Apply final scaling
        processed_data = self._apply_final_scaling(processed_data)
        
        # Store final selected features
        self.selected_features = processed_data.columns.tolist()
        self.is_fitted = True
        
        # Log final results
        self._log_feature_evolution("output", processed_data)
        
        logger.info("Feature processing pipeline completed:")
        logger.info(f"  - Input features: {self.feature_evolution['input']['n_features']}")
        logger.info(f"  - Output features: {processed_data.shape[1]}")
        logger.info(f"  - Reduction ratio: {processed_data.shape[1] / self.feature_evolution['input']['n_features']:.2f}")
        logger.info(f"  - Pipeline steps: {' -> '.join(self.pipeline_steps)}")
        
        return processed_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            data: New dataframe to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        logger.info(f"Transforming new data: {data.shape}")
        
        processed_data = data.copy()
        
        # Apply same pipeline steps
        if self.create_technical_indicators:
            processed_data = self.technical_engine.transform(processed_data)
            
        if self.imputer is not None:
            imputed_array = self.imputer.transform(processed_data)
            processed_data = pd.DataFrame(
                imputed_array,
                index=processed_data.index,
                columns=processed_data.columns
            )
            
        if self.low_variance_features:
            processed_data = processed_data.drop(columns=self.low_variance_features, errors='ignore')
            
        if self.high_correlation_features:
            processed_data = processed_data.drop(columns=self.high_correlation_features, errors='ignore')
            
        if self.pca_reducer is not None:
            processed_data = self.pca_reducer.transform(processed_data)
            
        if self.final_scaler is not None:
            scaled_array = self.final_scaler.transform(processed_data)
            processed_data = pd.DataFrame(
                scaled_array,
                index=processed_data.index,
                columns=processed_data.columns
            )
        
        logger.info(f"Transform completed: {processed_data.shape}")
        return processed_data
    
    def get_feature_importance_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze feature importance throughout pipeline.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with feature analysis
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before analysis")
            
        analysis = {
            'pipeline_steps': self.pipeline_steps,
            'feature_evolution': self.feature_evolution,
            'feature_reduction_summary': {
                'original_features': self.feature_evolution['input']['n_features'],
                'final_features': len(self.selected_features),
                'reduction_ratio': len(self.selected_features) / self.feature_evolution['input']['n_features'],
                'reduction_percentage': (1 - len(self.selected_features) / self.feature_evolution['input']['n_features']) * 100
            }
        }
        
        # Add PCA analysis if available
        if self.pca_reducer is not None:
            pca_results = self.pca_reducer.benchmark_inference_speed()
            analysis['pca_performance'] = pca_results
            
        # Add technical indicator summary if available
        if self.technical_engine is not None:
            analysis['technical_indicators'] = self.technical_engine.get_feature_summary()
            
        return analysis
    
    def get_pipeline_summary(self) -> pd.DataFrame:
        """
        Get summary of pipeline steps and their impact.
        
        Returns:
            DataFrame with pipeline step summary
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting summary")
            
        summary_data = []
        prev_features = self.feature_evolution['input']['n_features']
        
        for step in self.pipeline_steps:
            step_info = self.feature_evolution[step]
            features_removed = prev_features - step_info['n_features']
            
            summary_data.append({
                'step': step,
                'features_before': prev_features,
                'features_after': step_info['n_features'],
                'features_removed': features_removed,
                'removal_percentage': features_removed / prev_features * 100 if prev_features > 0 else 0,
                'missing_percentage': step_info['missing_percentage'],
                'memory_usage_mb': step_info['memory_usage_mb']
            })
            
            prev_features = step_info['n_features']
            
        return pd.DataFrame(summary_data)
