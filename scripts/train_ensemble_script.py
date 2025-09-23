import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.loader import HFTDataLoader
from src.features.pca_reducer import PCADimensionalityReducer
from src.models.autoencoder_mlp import AutoencoderMLP
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble_model import AutoencoderXGBoostEnsemble
from src.evaluation.utility_metric import JaneStreetUtilityMetric
from src.utils.config import get_config
from src.utils.timer import PerformanceTimer, timed_operation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_samples: int = 50000, n_features: int = 130) -> pd.DataFrame:
    """Create synthetic Jane Street-like data for demonstration."""
    np.random.seed(42)
    
    logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features")
    
    data = pd.DataFrame({
        'date': np.random.randint(0, 500, n_samples),  # ~500 trading days
        'ts_id': range(n_samples),
        'weight': np.random.uniform(0.1, 2.0, n_samples),
        'resp': np.random.normal(0, 0.01, n_samples),  # Small returns
    })
    
    # Add correlated feature columns (simulate real market data structure)
    base_factors = np.random.normal(0, 1, (n_samples, 20))  # 20 base factors
    
    for i in range(n_features):
        # Each feature is a combination of base factors plus noise
        factor_weights = np.random.normal(0, 0.5, 20)
        feature_values = base_factors @ factor_weights + np.random.normal(0, 0.3, n_samples)
        data[f'feature_{i}'] = feature_values
    
    # Make some features more predictive
    for i in range(0, min(10, n_features)):
        data[f'feature_{i}'] += data['resp'] * np.random.uniform(5, 15)
    
    # Add some missing values
    missing_mask = np.random.random((n_samples, n_features)) < 0.02
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data[feature_cols] = data[feature_cols].where(~missing_mask, np.nan)
    
    return data


def main(args):
    """Main training and validation pipeline."""
    
    timer = PerformanceTimer()
    
    # Load configuration
    config = get_config()
    logger.info("Configuration loaded")
    
    # Load data
    logger.info("Loading data...")
    with timed_operation("data_loading", timer):
        try:
            # Try to load real data
            loader = HFTDataLoader(args.data_path)
            X, y, metadata = loader.load_processed_data()
            logger.info(f"Loaded real data from {args.data_path}")
            
        except FileNotFoundError:
            logger.warning(f"Data file not found: {args.data_path}")
            logger.info("Creating synthetic data for demonstration...")
            
            synthetic_data = create_synthetic_data(
                n_samples=args.n_samples, 
                n_features=args.n_features
            )
            
            # Save synthetic data temporarily
            temp_path = "temp_synthetic_data.csv"
            synthetic_data.to_csv(temp_path, index=False)
            
            # Load through normal pipeline
            loader = HFTDataLoader(temp_path)
            X, y, metadata = loader.load_processed_data()
            
            # Create synthetic weights and returns for utility calculation
            synthetic_data['weight'] = np.random.uniform(0.5, 2.0, len(synthetic_data))
            synthetic_data['resp'] = np.random.normal(0, 0.01, len(synthetic_data))
            
            # Cleanup
            Path(temp_path).unlink()
            
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create train/test split (time-series aware)
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # For synthetic data, create weights and returns
    if 'synthetic_data' in locals():
        weights = pd.Series(np.random.uniform(0.5, 2.0, len(X)), index=X.index)
        returns = pd.Series(np.random.normal(0, 0.01, len(X)), index=X.index)
    else:
        # Use actual weights and returns from data if available
        weights = loader.data.get('weight', pd.Series(np.ones(len(X)), index=X.index))
        returns = loader.data.get('resp', pd.Series(np.random.normal(0, 0.01, len(X)), index=X.index))
    
    weights_train, weights_test = weights.iloc[:split_point], weights.iloc[split_point:]
    returns_train, returns_test = returns.iloc[:split_point], returns.iloc[split_point:]
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Apply PCA dimensionality reduction
    logger.info("Applying PCA dimensionality reduction...")
    with timed_operation("pca_fitting", timer):
        pca_reducer = PCADimensionalityReducer(variance_threshold=0.95)
        X_train_reduced = pca_reducer.fit_transform(X_train)
        X_test_reduced = pca_reducer.transform(X_test)
        
    # Benchmark PCA performance
    pca_results = pca_reducer.benchmark_inference_speed(n_samples=1000, n_trials=5)
    
    logger.info(f"PCA completed:")
    logger.info(f"  Original dimensions: {pca_results['original_dimensions']}")
    logger.info(f"  Reduced dimensions: {pca_results['reduced_dimensions']}")
    logger.info(f"  Reduction: {pca_results['dimensionality_reduction_pct']:.1f}%")
    logger.info(f"  Speedup: {pca_results['speedup_factor']:.1f}x")
    
    # Train individual models
    models = {}
    
    # Train Autoencoder-MLP
    logger.info("Training Autoencoder-MLP...")
    with timed_operation("autoencoder_training", timer):
        autoencoder = AutoencoderMLP(
            encoding_dim=32,
            hidden_dims=[64, 32],
            autoencoder_epochs=30,
            mlp_epochs=50,
            batch_size=512
        )
        autoencoder.fit(X_train_reduced, y_train)
        models['autoencoder'] = autoencoder
    
    # Train XGBoost
    logger.info("Training XGBoost...")
    with timed_operation("xgboost_training", timer):
        xgboost = XGBoostModel(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=15
        )
        xgboost.fit(X_train_reduced, y_train)
        models['xgboost'] = xgboost
    
    # Train Ensemble
    logger.info("Training Ensemble...")
    with timed_operation("ensemble_training", timer):
        ensemble = AutoencoderXGBoostEnsemble(
            autoencoder_weight=0.3,
            xgboost_weight=0.7,
            optimize_weights=True
        )
        
        # Use pre-trained models if possible
        ensemble.autoencoder = autoencoder
        ensemble.xgboost = xgboost
        ensemble.is_fitted = True
        ensemble.feature_names = X_train_reduced.columns.tolist()
        
        models['ensemble'] = ensemble
    
    # Log training times
    logger.info("Training completed:")
    logger.info(f"  Autoencoder: {timer.get_timing('autoencoder_training'):.1f}s")
    logger.info(f"  XGBoost: {timer.get_timing('xgboost_training'):.1f}s")
    logger.info(f"  Ensemble: {timer.get_timing('ensemble_training'):.1f}s")
    
    # Save results if requested
    if args.save_results:
        results_dict = {
            'data_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'train_size': len(X_train),
                'test_size': len(X_test)
            },
            'pca_results': pca_results,
            'training_times': timer.get_all_timings(),
            'claims_validation': claims_results
        }
        
        import json
        with open('training_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results_dict, f, indent=2, default=convert_numpy)
            
        logger.info("Results saved to training_results.json")
    
    return models, claims_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Jane Street ensemble models")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/train.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=50000,
        help="Number of synthetic samples if real data not found"
    )
    parser.add_argument(
        "--n_features", 
        type=int, 
        default=130,
        help="Number of synthetic features if real data not found"
    )
    parser.add_argument(
        "--save_results", 
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        models, results = main(args)
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
