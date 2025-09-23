"""
Autoencoder-MLP model for Jane Street market prediction.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging
from sklearn.model_selection import train_test_split

from .base_model import BaseModel
from ..utils.config import get_config

logger = logging.getLogger(__name__)

# Set TensorFlow logging to reduce noise
tf.get_logger().setLevel('ERROR')


class AutoencoderMLP(BaseModel):
    """
    Autoencoder-MLP model that learns compressed representations then predicts.
    
    Architecture:
    1. Autoencoder: Learn compressed feature representation (unsupervised)
    2. MLP: Use encoded features for binary classification (supervised)
    """
    
    def __init__(self, 
                 encoding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 autoencoder_epochs: int = 50,
                 mlp_epochs: int = 100,
                 batch_size: int = 1024,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 random_state: int = 42):
        """
        Initialize Autoencoder-MLP model.
        
        Args:
            encoding_dim: Dimension of encoded features
            hidden_dims: Hidden layer dimensions for autoencoder
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            autoencoder_epochs: Training epochs for autoencoder
            mlp_epochs: Training epochs for MLP
            batch_size: Batch size for training
            validation_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            random_state: Random seed
        """
        super().__init__("AutoencoderMLP")
        
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.autoencoder_epochs = autoencoder_epochs
        self.mlp_epochs = mlp_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Model components
        self.autoencoder = None
        self.encoder = None
        self.mlp_classifier = None
        
        # Training history
        self.autoencoder_history = None
        self.mlp_history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_autoencoder(self, input_dim: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Build autoencoder and encoder models.
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            Tuple of (autoencoder, encoder) models
        """
        # Input layer
        input_layer = tf.keras.Input(shape=(input_dim,), name='encoder_input')
        
        # Encoder
        x = input_layer
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = tf.keras.layers.Dense(
                hidden_dim, 
                activation='relu',
                name=f'encoder_dense_{i}'
            )(x)
            x = tf.keras.layers.Dropout(self.dropout_rate, name=f'encoder_dropout_{i}')(x)
            
        # Encoded representation (bottleneck)
        encoded = tf.keras.layers.Dense(
            self.encoding_dim, 
            activation='relu',
            name='encoded'
        )(x)
        
        # Decoder
        x = encoded
        for i, hidden_dim in enumerate(reversed(self.hidden_dims)):
            x = tf.keras.layers.Dense(
                hidden_dim,
                activation='relu', 
                name=f'decoder_dense_{i}'
            )(x)
            x = tf.keras.layers.Dropout(self.dropout_rate, name=f'decoder_dropout_{i}')(x)
            
        # Output layer (reconstruction)
        decoded = tf.keras.layers.Dense(
            input_dim,
            activation='linear',
            name='decoder_output'
        )(x)
        
        # Create models
        autoencoder = tf.keras.Model(input_layer, decoded, name='autoencoder')
        encoder = tf.keras.Model(input_layer, encoded, name='encoder')
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder, encoder
    
    def _build_mlp_classifier(self, encoding_dim: int) -> tf.keras.Model:
        """
        Build MLP classifier on encoded features.
        
        Args:
            encoding_dim: Encoded feature dimension
            
        Returns:
            Compiled MLP classifier model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(encoding_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name='mlp_classifier')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoencoderMLP':
        """
        Train autoencoder then MLP classifier.
        
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
            
        logger.info(f"Training AutoencoderMLP on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy for TensorFlow
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)
        
        # Build models
        input_dim = X_array.shape[1]
        self.autoencoder, self.encoder = self._build_autoencoder(input_dim)
        self.mlp_classifier = self._build_mlp_classifier(self.encoding_dim)
        
        logger.info(f"Built autoencoder: {input_dim} -> {self.encoding_dim} -> {input_dim}")
        logger.info(f"Built MLP classifier: {self.encoding_dim} -> 1")
        
        # Phase 1: Train autoencoder (unsupervised)
        logger.info("Phase 1: Training autoencoder (unsupervised)")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=0
            )
        ]
        
        self.autoencoder_history = self.autoencoder.fit(
            X_array, X_array,  # Reconstruct input
            epochs=self.autoencoder_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        # Phase 2: Extract encoded features
        logger.info("Phase 2: Extracting encoded features")
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size)
        
        # Phase 3: Train MLP classifier (supervised)
        logger.info("Phase 3: Training MLP classifier (supervised)")
        
        self.mlp_history = self.mlp_classifier.fit(
            X_encoded, y_array,
            epochs=self.mlp_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        
        # Log training results
        ae_final_loss = self.autoencoder_history.history['val_loss'][-1]
        mlp_final_auc = self.mlp_history.history['val_auc'][-1]
        
        logger.info(f"Training complete:")
        logger.info(f"  - Autoencoder final validation loss: {ae_final_loss:.4f}")
        logger.info(f"  - MLP final validation AUC: {mlp_final_auc:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using autoencoder + MLP.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Prediction array (0/1 decisions)
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
        
        # Convert to numpy
        X_array = X[self.feature_names].values.astype(np.float32)
        
        # Encode features
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size)
        
        # Predict with MLP
        probabilities_pos = self.mlp_classifier.predict(X_encoded, batch_size=self.batch_size)
        probabilities_pos = probabilities_pos.flatten()
        
        # Return as [prob_class_0, prob_class_1] format
        probabilities = np.column_stack([1 - probabilities_pos, probabilities_pos])
        
        return probabilities
    
    def get_encoded_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded feature representation.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Encoded features dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
            
        self.validate_input(X)
        
        X_array = X[self.feature_names].values.astype(np.float32)
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size)
        
        # Create column names for encoded features
        encoded_columns = [f'encoded_{i}' for i in range(self.encoding_dim)]
        
        return pd.DataFrame(X_encoded, columns=encoded_columns, index=X.index)
    
    def get_reconstruction_error(self, X: pd.DataFrame) -> pd.Series:
        """
        Get reconstruction error for anomaly detection.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Reconstruction error per sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating reconstruction error")
            
        self.validate_input(X)
        
        X_array = X[self.feature_names].values.astype(np.float32)
        X_reconstructed = self.autoencoder.predict(X_array, batch_size=self.batch_size)
        
        # Calculate MSE per sample
        mse_per_sample = np.mean((X_array - X_reconstructed) ** 2, axis=1)
        
        return pd.Series(mse_per_sample, index=X.index)
    
    def get_training_history(self) -> Dict[str, Dict]:
        """
        Get training history for both autoencoder and MLP.
        
        Returns:
            Dictionary with training histories
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting training history")
            
        return {
            'autoencoder': self.autoencoder_history.history,
            'mlp': self.mlp_history.history
        }
    
    def get_model_summary(self) -> Dict[str, str]:
        """
        Get model architecture summaries.
        
        Returns:
            Dictionary with model summaries
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summaries")
            
        from io import StringIO
        
        # Capture model summaries
        summaries = {}
        
        for name, model in [('autoencoder', self.autoencoder), 
                           ('encoder', self.encoder),
                           ('mlp_classifier', self.mlp_classifier)]:
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summaries[name] = stream.getvalue()
            
        return summaries
