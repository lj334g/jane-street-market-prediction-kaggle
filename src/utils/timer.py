"""Performance timing utilities for benchmarking"""

import time
import functools
from typing import Dict, Any, Callable, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Timer for measuring and tracking performance metrics."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        
    def start(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = time.time()
        
    def stop(self, name: str) -> float:
        """
        Stop timing an operation and return duration.
        
        Args:
            name: Operation name
            
        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' not started")
            
        duration = time.time() - self.start_times[name]
        self.timings[name] = duration
        del self.start_times[name]
        
        logger.info(f"Timer '{name}': {duration:.4f} seconds")
        return duration
        
    def get_timing(self, name: str) -> Optional[float]:
        """Get timing for a specific operation."""
        return self.timings.get(name)
        
    def get_all_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()
        
    def clear(self) -> None:
        """Clear all timings."""
        self.timings.clear()
        self.start_times.clear()
        
    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing code blocks."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        
        # Add timing to result if it's a dictionary
        if isinstance(result, dict):
            result['_execution_time'] = duration
            
        return result
    
    return wrapper


class InferenceSpeedBenchmark:
    """
    Benchmark inference speed for dimensionality reduction claims.
    """
    
    def __init__(self, n_samples: int = 1000, n_trials: int = 5):
        """
        Initialize benchmark.
        
        Args:
            n_samples: Number of samples for inference
            n_trials: Number of trials for averaging
        """
        self.n_samples = n_samples
        self.n_trials = n_trials
        
    def benchmark_inference_speed(self, 
                                 original_dims: int, 
                                 reduced_dims: int) -> Dict[str, float]:
        """
        Benchmark inference speed for original vs reduced dimensions.
        
        Args:
            original_dims: Original feature dimensionality
            reduced_dims: Reduced feature dimensionality after PCA
            
        Returns:
            Dictionary with timing results and speedup factor
        """
        import numpy as np
        
        # Create random model weights for simulation
        original_weights = np.random.random((original_dims, 1))
        reduced_weights = np.random.random((reduced_dims, 1))
        
        # Benchmark original dimensions
        original_times = []
        for _ in range(self.n_trials):
            X_original = np.random.random((self.n_samples, original_dims))
            
            start_time = time.time()
            _ = X_original @ original_weights
            original_times.append(time.time() - start_time)
            
        # Benchmark reduced dimensions
        reduced_times = []
        for _ in range(self.n_trials):
            X_reduced = np.random.random((self.n_samples, reduced_dims))
            
            start_time = time.time()
            _ = X_reduced @ reduced_weights
            reduced_times.append(time.time() - start_time)
            
        # Calculate averages
        avg_original_time = sum(original_times) / len(original_times)
        avg_reduced_time = sum(reduced_times) / len(reduced_times)
        
        # Calculate speedup factor
        speedup_factor = avg_original_time / avg_reduced_time
        
        results = {
            'original_dimensions': original_dims,
            'reduced_dimensions': reduced_dims,
            'avg_original_inference_time': avg_original_time,
            'avg_reduced_inference_time': avg_reduced_time,
            'speedup_factor': speedup_factor,
            'dimensionality_reduction_pct': (original_dims - reduced_dims) / original_dims,
            'n_samples': self.n_samples,
            'n_trials': self.n_trials
        }
        
        logger.info(f"Inference Speed Benchmark Results:")
        logger.info(f"  Original dimensions: {original_dims}")
        logger.info(f"  Reduced dimensions: {reduced_dims}")
        logger.info(f"  Original inference time: {avg_original_time:.6f}s")
        logger.info(f"  Reduced inference time: {avg_reduced_time:.6f}s")
        logger.info(f"  Speedup factor: {speedup_factor:.2f}x")
        logger.info(f"  Dimensionality reduction: {results['dimensionality_reduction_pct']:.1%}")
        
        return results


# Global timer instance for convenience
global_timer = PerformanceTimer()


@contextmanager
def timed_operation(name: str, timer: Optional[PerformanceTimer] = None):
    """
    Context manager for timing operations.
    
    Args:
        name: Operation name
        timer: Timer instance (uses global if None)
    """
    if timer is None:
        timer = global_timer
        
    timer.start(name)
    try:
        yield timer
    finally:
        timer.stop(name)


def benchmark_model_performance(model_func: Callable, 
                              X_test, 
                              name: str = "model") -> Dict[str, Any]:
    """
    Benchmark model prediction performance.
    
    Args:
        model_func: Model prediction function
        X_test: Test data
        name: Benchmark name
        
    Returns:
        Performance metrics
    """
    n_samples = len(X_test)
    
    # Single prediction timing
    start_time = time.time()
    predictions = model_func(X_test)
    single_time = time.time() - start_time
    
    # Multiple runs for stability
    times = []
    for _ in range(5):
        start_time = time.time()
        _ = model_func(X_test)
        times.append(time.time() - start_time)
        
    avg_time = sum(times) / len(times)
    
    results = {
        'model_name': name,
        'n_samples': n_samples,
        'single_prediction_time': single_time,
        'avg_prediction_time': avg_time,
        'predictions_per_second': n_samples / avg_time,
        'time_per_sample': avg_time / n_samples,
        'predictions_shape': predictions.shape if hasattr(predictions, 'shape') else len(predictions)
    }
    
    logger.info(f"Model Performance Benchmark - {name}:")
    logger.info(f"  Samples: {n_samples:,}")
    logger.info(f"  Avg prediction time: {avg_time:.4f}s")
    logger.info(f"  Predictions/second: {results['predictions_per_second']:.0f}")
    logger.info(f"  Time per sample: {results['time_per_sample']:.6f}s")
    
    return results
