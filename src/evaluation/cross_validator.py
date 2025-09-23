"""
Time-series aware cross-validation for financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Generator
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from .utility_metric import JaneStreetUtilityMetric

logger = logging.getLogger(__name__)


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validation with purging to prevent data leakage.
    Designed for financial time series with overlapping samples.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 max_train_group_size: Optional[int] = None,
                 gap_size: int = 0,
                 group_column: str = 'date'):
        """
        Initialize purged group time series split.
        
        Args:
            n_splits: Number of splits
            max_train_group_size: Maximum size for training groups
            gap_size: Gap between train and test (purging period)
            group_column: Column name for grouping (e.g., 'date')
        """
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.gap_size = gap_size
        self.group_column = group_column
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Generator:
        """
        Generate train/test splits with purging.
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            groups: Group series (optional, will use group_column from X if None)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            if self.group_column not in X.columns:
                raise ValueError(f"Group column '{self.group_column}' not found in data")
            groups = X[self.group_column]
            
        # Get unique groups and sort them
        unique_groups = sorted(groups.unique())
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits + 1:
            raise ValueError(f"Not enough groups ({n_groups}) for {self.n_splits} splits")
            
        # Calculate group splits
        group_splits = np.array_split(unique_groups, self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Training groups: all groups up to split i
            train_groups = []
            for j in range(i + 1):
                train_groups.extend(group_splits[j])
                
            # Test groups: next group split
            test_groups = group_splits[i + 1]
            
            # Apply max train group size if specified
            if self.max_train_group_size is not None and len(train_groups) > self.max_train_group_size:
                train_groups = train_groups[-self.max_train_group_size:]
                
            # Apply gap (purging) - remove groups close to test period
            if self.gap_size > 0:
                gap_start = min(test_groups) - self.gap_size
                train_groups = [g for g in train_groups if g < gap_start]
                
            # Get indices
            train_mask = groups.isin(train_groups)
            test_mask = groups.isin(test_groups)
            
            train_indices = X.index[train_mask].tolist()
            test_indices = X.index[test_mask].tolist()
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
                
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class FinancialCrossValidator:
    """
    Cross-validator specifically designed for financial time series.
    Implements multiple CV strategies and utility-based evaluation.
    """
    
    def __init__(self,
                 cv_strategy: str = 'purged_group_ts',
                 n_splits: int = 5,
                 gap_size: int = 0,
                 group_column: str = 'date',
                 utility_metric: Optional[JaneStreetUtilityMetric] = None,
                 random_state: int = 42):
        """
        Initialize financial cross-validator.
        
        Args:
            cv_strategy: Cross-validation strategy ('purged_group_ts', 'time_series', 'stratified')
            n_splits: Number of splits
            gap_size: Gap between train/test for purging
            group_column: Column for grouping
            utility_metric: Jane Street utility metric instance
            random_state: Random seed
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.gap_size = gap_size
        self.group_column = group_column
        self.utility_metric = utility_metric or JaneStreetUtilityMetric()
        self.random_state = random_state
        
        # Initialize cross-validator
        self.cv_splitter = self._create_cv_splitter()
        
    def _create_cv_splitter(self) -> BaseCrossValidator:
        """Create cross-validation splitter based on strategy."""
        if self.cv_strategy == 'purged_group_ts':
            return PurgedGroupTimeSeriesSplit(
                n_splits=self.n_splits,
                gap_size=self.gap_size,
                group_column=self.group_column
            )
        elif self.cv_strategy == 'time_series':
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=self.n_splits)
        elif self.cv_strategy == 'stratified':
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def cross_validate_model(self,
                           model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           weights: Optional[pd.Series] = None,
                           returns: Optional[pd.Series] = None,
                           scoring_metrics: List[str] = ['utility', 'auc', 'accuracy']) -> Dict[str, Any]:
        """
        Cross-validate model with financial-specific metrics.
        
        Args:
            model: Model instance with fit/predict_proba methods
            X: Feature dataframe
            y: Target series
            weights: Sample weights for utility calculation
            returns: Sample returns for utility calculation
            scoring_metrics: List of metrics to compute
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Cross-validating model with {self.cv_strategy} strategy ({self.n_splits} folds)")
        
        # Initialize results storage
        fold_results = []
        fold_predictions = []
        
        # Perform cross-validation
        fold_num = 0
        for train_idx, test_idx in self.cv_splitter.split(X, y):
            logger.debug(f"Processing fold {fold_num + 1}/{self.n_splits}")
            
            # Split data
            if isinstance(train_idx, list):
                X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train, y_test = y.loc[train_idx], y.loc[test_idx]
                if weights is not None:
                    weights_test = weights.loc[test_idx]
                if returns is not None:
                    returns_test = returns.loc[test_idx]
            else:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if weights is not None:
                    weights_test = weights.iloc[test_idx]
                if returns is not None:
                    returns_test = returns.iloc[test_idx]
            
            # Train model
            model_fold = self._clone_model(model)
            model_fold.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(model_fold, 'predict_proba'):
                y_pred_proba = model_fold.predict_proba(X_test)
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # Positive class probabilities
            else:
                y_pred_proba = model_fold.predict(X_test).astype(float)
                
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            fold_metrics = {}
            
            # Standard ML metrics
            if 'auc' in scoring_metrics:
                try:
                    fold_metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    fold_metrics['auc'] = 0.5  # If only one class present
                    
            if 'accuracy' in scoring_metrics:
                fold_metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
                
            if 'precision' in scoring_metrics:
                fold_metrics['precision'] = precision_score(y_test, y_pred_binary, zero_division=0)
                
            if 'recall' in scoring_metrics:
                fold_metrics['recall'] = recall_score(y_test, y_pred_binary, zero_division=0)
            
            # Utility metric (if data available)
            if 'utility' in scoring_metrics and weights is not None and returns is not None:
                fold_metrics['utility'] = self.utility_metric.calculate_utility(
                    y_test.values, y_pred_proba, weights_test.values, returns_test.values
                )
            
            # Store fold results
            fold_results.append({
                'fold': fold_num,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': fold_metrics
            })
            
            # Store predictions for later analysis
            fold_predictions.append({
                'fold': fold_num,
                'test_indices': test_idx,
                'y_true': y_test.values,
                'y_pred_proba': y_pred_proba,
                'y_pred_binary': y_pred_binary
            })
            
            fold_num += 1
            
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results, fold_predictions, scoring_metrics)
        
        logger.info("Cross-validation completed:")
        for metric in scoring_metrics:
            if metric in cv_results['mean_scores']:
                logger.info(f"  - {metric}: {cv_results['mean_scores'][metric]:.4f} ± {cv_results['std_scores'][metric]:.4f}")
        
        return cv_results
    
    def _clone_model(self, model: Any) -> Any:
        """Clone model for cross-validation."""
        # Try to use sklearn clone if available
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback: create new instance with same parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                return model.__class__(**params)
            else:
                # Last resort: return same instance (not ideal but works for some models)
                return model
    
    def _aggregate_cv_results(self, fold_results: List[Dict], fold_predictions: List[Dict], 
                            scoring_metrics: List[str]) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        
        # Extract metrics from fold results
        all_metrics = {metric: [] for metric in scoring_metrics}
        
        for fold_result in fold_results:
            for metric in scoring_metrics:
                if metric in fold_result['metrics']:
                    all_metrics[metric].append(fold_result['metrics'][metric])
        
        # Calculate mean and std for each metric
        mean_scores = {}
        std_scores = {}
        
        for metric, values in all_metrics.items():
            if values:  # If metric was calculated
                mean_scores[metric] = np.mean(values)
                std_scores[metric] = np.std(values)
            else:
                mean_scores[metric] = np.nan
                std_scores[metric] = np.nan
        
        # Compile overall results
        cv_results = {
            'cv_strategy': self.cv_strategy,
            'n_splits': self.n_splits,
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'individual_scores': all_metrics,
            'fold_results': fold_results,
            'fold_predictions': fold_predictions
        }
        
        return cv_results
    
    def compare_models(self,
                      models: Dict[str, Any],
                      X: pd.DataFrame,
                      y: pd.Series,
                      weights: Optional[pd.Series] = None,
                      returns: Optional[pd.Series] = None,
                      scoring_metrics: List[str] = ['utility', 'auc']) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models: Dictionary mapping model names to model instances
            X: Feature dataframe
            y: Target series
            weights: Sample weights
            returns: Sample returns
            scoring_metrics: Metrics to compare
            
        Returns:
            Dictionary with model comparison results
        """
        logger.info(f"Comparing {len(models)} models with cross-validation")
        
        model_results = {}
        
        # Cross-validate each model
        for model_name, model in models.items():
            logger.info(f"Cross-validating {model_name}...")
            
            model_cv_results = self.cross_validate_model(
                model, X, y, weights, returns, scoring_metrics
            )
            
            model_results[model_name] = model_cv_results
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(model_results, scoring_metrics)
        
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(model_results, scoring_metrics)
        
        comparison_results = {
            'model_results': model_results,
            'comparison_summary': comparison_summary,
            'significance_tests': significance_tests,
            'best_models': self._identify_best_models(comparison_summary, scoring_metrics)
        }
        
        # Log comparison results
        logger.info("Model comparison results:")
        for metric in scoring_metrics:
            if metric in comparison_summary.columns:
                logger.info(f"\n{metric.upper()} scores:")
                for model_name in comparison_summary.index:
                    score = comparison_summary.loc[model_name, metric]
                    logger.info(f"  - {model_name}: {score:.4f}")
        
        return comparison_results
    
    def _create_comparison_summary(self, model_results: Dict[str, Dict], scoring_metrics: List[str]) -> pd.DataFrame:
        """Create comparison summary dataframe."""
        summary_data = []
        
        for model_name, results in model_results.items():
            row_data = {'model': model_name}
            
            for metric in scoring_metrics:
                if metric in results['mean_scores']:
                    row_data[metric] = results['mean_scores'][metric]
                    row_data[f'{metric}_std'] = results['std_scores'][metric]
                    
            summary_data.append(row_data)
        
        summary_df = pd.DataFrame(summary_data).set_index('model')
        return summary_df
    
    def _perform_significance_tests(self, model_results: Dict[str, Dict], scoring_metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        from scipy import stats
        
        significance_tests = {}
        model_names = list(model_results.keys())
        
        for metric in scoring_metrics:
            metric_tests = {}
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    
                    scores1 = model_results[model1]['individual_scores'].get(metric, [])
                    scores2 = model_results[model2]['individual_scores'].get(metric, [])
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        
                        metric_tests[f'{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_at_05': p_value < 0.05,
                            'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2
                        }
            
            significance_tests[metric] = metric_tests
        
        return significance_tests
    
    def _identify_best_models(self, comparison_summary: pd.DataFrame, scoring_metrics: List[str]) -> Dict[str, str]:
        """Identify best performing model for each metric."""
        best_models = {}
        
        for metric in scoring_metrics:
            if metric in comparison_summary.columns:
                # Higher is better for most metrics
                if metric == 'utility':
                    # For utility, higher absolute value might be better (could be negative)
                    best_idx = comparison_summary[metric].abs().idxmax()
                else:
                    # For AUC, accuracy, etc., higher is better
                    best_idx = comparison_summary[metric].idxmax()
                    
                best_models[metric] = best_idx
        
        return best_models
    
    def validate_utility_improvement(self,
                                   baseline_model: Any,
                                   improved_model: Any,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   weights: pd.Series,
                                   returns: pd.Series,
                                   target_improvement_pct: float = 12.0) -> Dict[str, Any]:
        """
        Validate utility improvement claim through cross-validation.
        
        Args:
            baseline_model: Baseline model
            improved_model: Improved model to compare
            X: Feature dataframe
            y: Target series
            weights: Sample weights
            returns: Sample returns
            target_improvement_pct: Target improvement percentage
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {target_improvement_pct}% utility improvement claim...")
        
        # Compare models
        models = {
            'baseline': baseline_model,
            'improved': improved_model
        }
        
        comparison_results = self.compare_models(
            models, X, y, weights, returns, scoring_metrics=['utility']
        )
        
        # Extract utility scores
        baseline_utility = comparison_results['model_results']['baseline']['mean_scores']['utility']
        improved_utility = comparison_results['model_results']['improved']['mean_scores']['utility']
        
        # Calculate improvement
        improvement_pct = self.utility_metric.calculate_utility_improvement(baseline_utility, improved_utility)
        
        # Statistical significance
        significance_test = comparison_results['significance_tests']['utility'].get('baseline_vs_improved', {})
        
        validation_results = {
            'baseline_utility_mean': baseline_utility,
            'improved_utility_mean': improved_utility,
            'improvement_percentage': improvement_pct,
            'target_improvement': target_improvement_pct,
            'meets_target': improvement_pct >= target_improvement_pct,
            'statistical_significance': significance_test,
            'cv_details': {
                'baseline_scores': comparison_results['model_results']['baseline']['individual_scores']['utility'],
                'improved_scores': comparison_results['model_results']['improved']['individual_scores']['utility']
            }
        }
        
        logger.info("Utility improvement validation results:")
        logger.info(f"  - Baseline utility: {baseline_utility:.6f}")
        logger.info(f"  - Improved utility: {improved_utility:.6f}")
        logger.info(f"  - Improvement: {improvement_pct:.1f}%")
        logger.info(f"  - Meets {target_improvement_pct}% target: {validation_results['meets_target']}")
        logger.info(f"  - Statistically significant: {significance_test.get('significant_at_05', False)}")
        
        return validation_results
