"""
Jane Street competition utility metric calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)


class JaneStreetUtilityMetric:
    """
    Jane Street competition utility metric calculator.
    
    The utility function penalizes overtrading and rewards profitable predictions:
    utility = sum(returns * weights * actions) where actions are trading decisions
    """
    
    def __init__(self, 
                 decision_threshold: float = 0.5,
                 risk_penalty: float = 0.0):
        """
        Initialize utility metric calculator.
        
        Args:
            decision_threshold: Threshold for converting probabilities to decisions
            risk_penalty: Additional risk penalty factor (if needed)
        """
        self.decision_threshold = decision_threshold
        self.risk_penalty = risk_penalty
        
    def calculate_utility(self, 
                         y_true: np.ndarray,
                         y_pred: np.ndarray, 
                         weights: np.ndarray,
                         returns: np.ndarray,
                         convert_to_decisions: bool = True) -> float:
        """
        Calculate Jane Street utility metric.
        
        Args:
            y_true: True binary labels (not used in utility, but kept for interface)
            y_pred: Predicted probabilities or binary decisions
            weights: Sample weights from competition
            returns: Sample returns from competition
            convert_to_decisions: Whether to convert probabilities to binary decisions
            
        Returns:
            Utility score
        """
        # Convert probabilities to binary decisions if needed
        if convert_to_decisions and np.any((y_pred > 0) & (y_pred < 1)):
            actions = (y_pred >= self.decision_threshold).astype(int)
        else:
            actions = y_pred.astype(int)
            
        # Calculate utility: sum of weighted returns for positive actions
        utility = np.sum(returns * weights * actions)
        
        # Apply risk penalty if specified
        if self.risk_penalty > 0:
            n_trades = np.sum(actions)
            utility -= self.risk_penalty * n_trades
            
        return utility
    
    def calculate_utility_improvement(self, 
                                    baseline_utility: float, 
                                    model_utility: float) -> float:
        """
        Calculate percentage improvement over baseline.
        
        Args:
            baseline_utility: Baseline model utility
            model_utility: New model utility
            
        Returns:
            Improvement percentage
        """
        if baseline_utility == 0:
            return 100.0 if model_utility > 0 else 0.0
            
        improvement = (model_utility - baseline_utility) / abs(baseline_utility) * 100
        return improvement
    
    def create_sklearn_scorer(self, 
                            weights: np.ndarray, 
                            returns: np.ndarray) -> Any:
        """
        Create scikit-learn compatible scorer for cross-validation.
        
        Args:
            weights: Sample weights
            returns: Sample returns
            
        Returns:
            Scorer function compatible with sklearn cross-validation
        """
        def utility_scorer(y_true, y_pred):
            return self.calculate_utility(y_true, y_pred, weights, returns)
            
        return make_scorer(utility_scorer, greater_is_better=True, needs_proba=False)
    
    def cross_validate_utility(self,
                              model: Any,
                              X: pd.DataFrame,
                              y: pd.Series,
                              weights: pd.Series,
                              returns: pd.Series,
                              cv: Union[int, Any] = 5,
                              use_time_series_cv: bool = True) -> Dict[str, Any]:
        """
        Cross-validate utility metric.
        
        Args:
            model: Model to evaluate (must have predict_proba method)
            X: Feature dataframe
            y: Target series
            weights: Weight series
            returns: Returns series
            cv: Cross-validation strategy or number of folds
            use_time_series_cv: Whether to use time series aware CV
            
        Returns:
            Dictionary with cross-validation results
        """
        # Setup cross-validation
        if isinstance(cv, int):
            if use_time_series_cv:
                # Time series split to prevent look-ahead bias
                cv_splitter = TimeSeriesSplit(n_splits=cv)
                logger.info(f"Using TimeSeriesSplit with {cv} splits")
            else:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                logger.info(f"Using StratifiedKFold with {cv} splits")
        else:
            cv_splitter = cv
            
        # Manual cross-validation to handle sample-specific weights and returns
        utilities = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            weights_test = weights.iloc[test_idx]
            returns_test = returns.iloc[test_idx]
            
            # Train model
            model_fold = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            model_fold.fit(X_train, y_train)
            
            # Predict
            if hasattr(model_fold, 'predict_proba'):
                y_pred = model_fold.predict_proba(X_test)[:, 1]
            else:
                y_pred = model_fold.predict(X_test)
                
            # Calculate utility for this fold
            fold_utility = self.calculate_utility(
                y_test.values, y_pred, weights_test.values, returns_test.values
            )
            
            utilities.append(fold_utility)
            
            fold_results.append({
                'fold': fold,
                'utility': fold_utility,
                'n_samples': len(test_idx),
                'n_positive_actions': np.sum((y_pred >= self.decision_threshold).astype(int)),
                'mean_return': returns_test.mean(),
                'mean_weight': weights_test.mean()
            })
            
            logger.debug(f"Fold {fold}: utility={fold_utility:.6f}, samples={len(test_idx)}")
        
        # Aggregate results
        results = {
            'mean_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
            'utilities': utilities,
            'fold_results': fold_results,
            'cv_type': type(cv_splitter).__name__,
            'n_folds': len(utilities)
        }
        
        logger.info(f"Cross-validation complete:")
        logger.info(f"  - Mean utility: {results['mean_utility']:.6f}")
        logger.info(f"  - Std utility: {results['std_utility']:.6f}")
        
        return results
    
    def compare_models_cv(self,
                         model1: Any,
                         model2: Any,
                         X: pd.DataFrame,
                         y: pd.Series,
                         weights: pd.Series,
                         returns: pd.Series,
                         model1_name: str = "Model1",
                         model2_name: str = "Model2",
                         cv: Union[int, Any] = 5) -> Dict[str, Any]:
        """
        Compare two models using cross-validation.
        
        Args:
            model1: First model (baseline)
            model2: Second model (new model)
            X: Feature dataframe
            y: Target series
            weights: Weight series
            returns: Returns series
            model1_name: Name for first model
            model2_name: Name for second model
            cv: Cross-validation strategy
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}")
        
        # Cross-validate both models
        results1 = self.cross_validate_utility(model1, X, y, weights, returns, cv)
        results2 = self.cross_validate_utility(model2, X, y, weights, returns, cv)
        
        # Calculate improvement
        improvement = self.calculate_utility_improvement(
            results1['mean_utility'], results2['mean_utility']
        )
        
        # Statistical significance test (paired t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(results2['utilities'], results1['utilities'])
        
        comparison = {
            f'{model1_name}_results': results1,
            f'{model2_name}_results': results2,
            'utility_improvement': {
                'absolute': results2['mean_utility'] - results1['mean_utility'],
                'percentage': improvement,
                'meets_12_percent_target': improvement >= 12.0
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_05': p_value < 0.05
            },
            'summary': {
                f'{model1_name}_mean_utility': results1['mean_utility'],
                f'{model2_name}_mean_utility': results2['mean_utility'],
                'improvement_pct': improvement,
                'is_significant': p_value < 0.05
            }
        }
        
        logger.info(f"Model comparison results:")
        logger.info(f"  - {model1_name} utility: {results1['mean_utility']:.6f}")
        logger.info(f"  - {model2_name} utility: {results2['mean_utility']:.6f}")
        logger.info(f"  - Improvement: {improvement:.1f}%")
        logger.info(f"  - Significant at p<0.05: {p_value < 0.05}")
        logger.info(f"  - Meets 12% target: {improvement >= 12.0}")
        
        return comparison
    
    def simulate_trading_performance(self,
                                   predictions: np.ndarray,
                                   actual_returns: np.ndarray,
                                   weights: np.ndarray,
                                   transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Simulate trading performance with transaction costs.
        
        Args:
            predictions: Predicted probabilities
            actual_returns: Actual returns
            weights: Sample weights
            transaction_cost: Transaction cost per trade (as fraction)
            
        Returns:
            Dictionary with trading performance metrics
        """
        # Convert predictions to trading decisions
        actions = (predictions >= self.decision_threshold).astype(int)
        
        # Calculate gross returns
        gross_returns = actual_returns * weights * actions
        gross_utility = np.sum(gross_returns)
        
        # Calculate transaction costs
        n_trades = np.sum(actions)
        total_transaction_costs = n_trades * transaction_cost * np.sum(np.abs(actual_returns * weights))
        
        # Net utility after costs
        net_utility = gross_utility - total_transaction_costs
        
        # Performance metrics
        performance = {
            'gross_utility': gross_utility,
            'transaction_costs': total_transaction_costs,
            'net_utility': net_utility,
            'n_trades': n_trades,
            'trade_rate': n_trades / len(predictions),
            'avg_return_per_trade': np.mean(gross_returns[actions == 1]) if n_trades > 0 else 0,
            'hit_rate': np.mean((actual_returns * actions) > 0) if n_trades > 0 else 0
        }
        
        return performance
    
    def analyze_trading_decisions(self,
                                predictions: np.ndarray,
                                actual_returns: np.ndarray,
                                weights: np.ndarray) -> pd.DataFrame:
        """
        Analyze trading decisions by prediction confidence.
        
        Args:
            predictions: Predicted probabilities
            actual_returns: Actual returns
            weights: Sample weights
            
        Returns:
            DataFrame with analysis by prediction quantiles
        """
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'prediction': predictions,
            'actual_return': actual_returns,
            'weight': weights,
            'decision': (predictions >= self.decision_threshold).astype(int)
        })
        
        # Add prediction quantiles
        df['pred_quantile'] = pd.qcut(df['prediction'], q=10, labels=False)
        
        # Analyze by quantile
        analysis = df.groupby('pred_quantile').agg({
            'prediction': ['mean', 'count'],
            'actual_return': ['mean', 'std'],
            'weight': 'mean',
            'decision': 'mean'
        }).round(4)
        
        # Flatten column names
        analysis.columns = ['_'.join(col) for col in analysis.columns]
        
        # Add utility calculation per quantile
        utilities = []
        for q in range(10):
            quantile_data = df[df['pred_quantile'] == q]
            if len(quantile_data) > 0:
                utility = self.calculate_utility(
                    None, quantile_data['decision'].values,
                    quantile_data['weight'].values, 
                    quantile_data['actual_return'].values,
                    convert_to_decisions=False
                )
                utilities.append(utility)
            else:
                utilities.append(0.0)
        
        analysis['utility'] = utilities
        
        return analysis
