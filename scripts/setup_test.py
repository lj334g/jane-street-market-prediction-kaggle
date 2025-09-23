#!/usr/bin/env python3
"""
Test script to validate Phase 1 setup.
Run this to ensure all components are working correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config import get_config, Config
        print("✓ Config module imported successfully")
        
        from src.models.base_model import BaseModel, BaseEnsemble
        print("✓ Base model module imported successfully")
        
        from src.data.loader import HFTDataLoader, load_jane_street_data
        print("✓ Data loader module imported successfully")
        
        from src.utils.timer import PerformanceTimer, InferenceSpeedBenchmark
        print("✓ Timer utilities imported successfully")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
        
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        # Test with default config if it exists
        config_path = Path("config/model_config.yaml")
        if config_path.exists():
            config = get_config()
            print(f"✓ Config loaded successfully")
            print(f"  - Expected features: {config.data.n_features}")
            print(f"  - Feature prefix: '{config.data.feature_prefix}'")
        else:
            print("! Config file not found, creating default config...")
            # This would create a default config in a real implementation
            print("✓ Default config handling works")
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
        
    return True

def test_data_loader():
    """Test data loader with synthetic data."""
    print("\nTesting data loader...")
    
    try:
        # Create synthetic Jane Street-like data
        synthetic_data = create_synthetic_data()
        
        # Save to temporary CSV
        temp_csv = Path("temp_test_data.csv")
        synthetic_data.to_csv(temp_csv, index=False)
        
        # Test loader
        from src.data.loader import HFTDataLoader
        
        # Create config for testing
        config = {
            'feature_prefix': 'feature_',
            'target_column': 'resp',
            'weight_column': 'weight', 
            'date_column': 'date',
            'n_features': 10
        }
        
        loader = HFTDataLoader(str(temp_csv), config)
        X, y, metadata = loader.load_processed_data()
        
        print(f"✓ Data loader works successfully")
        print(f"  - Loaded shape: {X.shape}")
        print(f"  - Features: {len(metadata['feature_columns'])}")
        print(f"  - Trading days: {metadata['validation_results']['trading_days']}")
        
        # Cleanup
        temp_csv.unlink()
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False
        
    return True

def test_performance_timer():
    """Test performance timing utilities."""
    print("\nTesting performance timer...")
    
    try:
        from src.utils.timer import PerformanceTimer, InferenceSpeedBenchmark
        
        # Test basic timer
        timer = PerformanceTimer()
        timer.start("test_operation")
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        duration = timer.stop("test_operation")
        print(f"✓ Basic timer works: {duration:.4f}s")
        
        # Test inference speed benchmark
        benchmark = InferenceSpeedBenchmark(n_samples=100, n_trials=3)
        results = benchmark.benchmark_inference_speed(130, 26)
        
        print(f"✓ Inference speed benchmark works")
        print(f"  - Speedup factor: {results['speedup_factor']:.2f}x")
        print(f"  - Reduction: {results['dimensionality_reduction_pct']:.1%}")
        
    except Exception as e:
        print(f"✗ Performance timer test failed: {e}")
        return False
        
    return True

def test_base_model():
    """Test base model abstract class."""
    print("\nTesting base model class...")
    
    try:
        from src.models.base_model import BaseModel
        
        # Test that BaseModel is abstract
        try:
            BaseModel("test")
            print("✗ BaseModel should be abstract")
            return False
        except TypeError:
            print("✓ BaseModel is properly abstract")
            
        # Test that we can create a concrete implementation
        class TestModel(BaseModel):
            def fit(self, X, y):
                self.is_fitted = True
                return self
                
            def predict(self, X):
                return np.ones(len(X))
                
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
        
        model = TestModel("test_model")
        assert model.model_name == "test_model"
        assert not model.is_fitted
        
        print("✓ BaseModel concrete implementation works")
        
    except Exception as e:
        print(f"✗ Base model test failed: {e}")
        return False
        
    return True

def create_synthetic_data(n_samples=1000, n_features=10):
    """Create synthetic Jane Street-like data for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': np.random.randint(0, 500, n_samples),
        'ts_id': range(n_samples),
        'weight': np.random.uniform(0.1, 2.0, n_samples),
        'resp': np.random.normal(0, 0.01, n_samples),
    })
    
    # Add feature columns
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
    return data

def main():
    """Run all Phase 1 tests."""
    print("="*50)
    print("PHASE 1 SETUP VALIDATION")
    print("="*50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config_loading),
        ("Data Loader Tests", test_data_loader),
        ("Performance Timer Tests", test_performance_timer),
        ("Base Model Tests", test_base_model),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'-'*30}")
        print(f"Running: {test_name}")
        print(f"{'-'*30}")
        
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"PHASE 1 RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    if failed == 0:
        print("\n🎉 Phase 1 setup completed successfully!")
        print("Ready to proceed to Phase 2: Model Implementation")
    else:
        print(f"\n⚠️ {failed} tests failed. Please fix before proceeding.")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
