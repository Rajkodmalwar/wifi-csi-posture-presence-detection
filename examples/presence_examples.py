"""
Presence Detection Examples

5 educational examples demonstrating the presence detection system:
1. Load and explore WiFi CSI data
2. Expand features with statistical measures
3. Train and tune Random Forest model
4. Save and load model from disk
5. Make predictions with probability
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PresenceDetectionModel
from src.data_utils import DataValidator, DataStatistics
from src.utils import setup_logging
from src.config import PRESENCE_MODEL_CONFIG

# Setup logging
logger = setup_logging(level='INFO')


def example_1_load_and_explore():
    """Example 1: Load and explore WiFi CSI presence data."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Load and Explore WiFi CSI Data")
    logger.info("="*60)
    
    # Create synthetic WiFi CSI data for demo
    logger.info("Creating synthetic WiFi CSI data...")
    
    data = {
        'rssi': np.random.uniform(-80, -30, 200),
        'rate': np.random.choice([54, 54, 54, 54, 54, 54, 54, 54, 39], 200),
        'noise_floor': np.random.uniform(-95, -85, 200),
        'channel': np.random.choice([1, 6, 11], 200),
        'label': np.random.choice(['present', 'absent'], 200)
    }
    
    df = pd.DataFrame(data)
    
    logger.info(f"\nLoaded data shape: {df.shape}")
    logger.info(f"Features: {list(df.columns[:-1])}")
    logger.info(f"Classes: {df['label'].unique()}")
    logger.info(f"\nData sample:\n{df.head()}")
    
    # Data statistics
    logger.info(f"\nFeature statistics:")
    stats_df = DataStatistics.compute_feature_statistics(
        df[['rssi', 'rate', 'noise_floor', 'channel']].values,
        feature_names=['rssi', 'rate', 'noise_floor', 'channel']
    )
    logger.info(f"{stats_df}")
    
    return df[['rssi', 'rate', 'noise_floor', 'channel']].values, df['label'].values


def example_2_expand_features():
    """Example 2: Expand features with statistical measures."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Expand Features")
    logger.info("="*60)
    
    # Load data
    X, y = example_1_load_and_explore()
    
    # Initialize model and expand features
    model = PresenceDetectionModel()
    
    logger.info(f"Original features shape: {X.shape}")
    
    X_expanded = model.expand_features(X)
    
    logger.info(f"Expanded features shape: {X_expanded.shape}")
    logger.info(f"\nExpansion includes:")
    logger.info(f"  - Original features: {X.shape[1]}")
    logger.info(f"  - Squared features: {X.shape[1]}")
    
    # Calculate interaction terms
    interaction_count = (X.shape[1] * (X.shape[1] - 1)) // 2
    logger.info(f"  - Interaction terms: {interaction_count}")
    logger.info(f"  - Total: {X_expanded.shape[1]}")


def example_3_train_and_tune():
    """Example 3: Train and tune Random Forest model."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Train and Tune Model")
    logger.info("="*60)
    
    # Load data
    X, y = example_1_load_and_explore()
    
    # Initialize model
    model = PresenceDetectionModel()
    
    logger.info("Training with GridSearchCV hyperparameter tuning...")
    logger.info(f"  CV Folds: {PRESENCE_MODEL_CONFIG['cv_folds']}")
    logger.info(f"  Hyperparameter grid: {PRESENCE_MODEL_CONFIG['param_grid']}")
    
    # Train with tuning
    metrics = model.train_with_tuning(
        X=X,
        y=y,
        test_size=0.2,
        cv_folds=3  # Reduced for demo
    )
    
    logger.info(f"\nModel Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    
    return model, X, y


def example_4_save_load_model():
    """Example 4: Save and load model from disk."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Save and Load Model")
    logger.info("="*60)
    
    # Train model
    model, X, y = example_3_train_and_tune()
    
    # Save model
    save_dir = './demo_models'
    logger.info(f"Saving model to {save_dir}...")
    model.save_model(save_dir)
    
    # Load model
    logger.info(f"Loading model from {save_dir}...")
    
    loaded_model = PresenceDetectionModel()
    loaded_model.load_model(save_dir)
    
    logger.info("Model loaded successfully!")
    logger.info(f"  Model: {type(loaded_model.model).__name__}")
    logger.info(f"  Scaler: {type(loaded_model.scaler).__name__}")
    logger.info(f"  Label Encoder classes: {loaded_model.label_encoder.classes_}")
    
    return loaded_model, X


def example_5_predict_with_probability():
    """Example 5: Make predictions with probability."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Predict with Probability")
    logger.info("="*60)
    
    # Load trained model
    loaded_model, X = example_4_save_load_model()
    
    # Make predictions
    logger.info("Making predictions on new samples...")
    
    # Single predictions
    predictions = loaded_model.predict(X[:5])
    logger.info(f"\nSingle predictions (first 5):")
    for i, pred in enumerate(predictions):
        logger.info(f"  Sample {i}: {pred}")
    
    # Probability predictions
    probabilities = loaded_model.predict_proba(X[:5])
    logger.info(f"\nPrediction probabilities (first 5):")
    for i in range(min(5, len(probabilities))):
        logger.info(f"  Sample {i}:")
        for j, class_name in enumerate(loaded_model.label_encoder.classes_):
            logger.info(f"    {class_name}: {probabilities[i][j]:.4f}")


if __name__ == '__main__':
    """Run all examples."""
    
    logger.info("\n\n")
    logger.info("#"*60)
    logger.info("# WiFi CSI Presence Detection - Educational Examples")
    logger.info("#"*60)
    
    try:
        example_1_load_and_explore()
        example_2_expand_features()
        example_3_train_and_tune()
        example_4_save_load_model()
        example_5_predict_with_probability()
        
        logger.info("\n\n")
        logger.info("#"*60)
        logger.info("# All Examples Completed Successfully!")
        logger.info("#"*60)
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}", exc_info=True)
