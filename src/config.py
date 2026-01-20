"""
Configuration Module

Centralized configuration for WiFi CSI detection system.
Contains all hyperparameters, model architecture, and hardware settings.

Configuration categories:
    - CSI parameters: Hardware and signal processing config
    - Model architecture: CNN and RF parameters
    - Training parameters: Learning rates, epochs, batch sizes
    - Data parameters: Train/test split, normalization
    - Output paths: Model and checkpoint directories
"""

from typing import Dict, List
import logging


# ===== CSI HARDWARE PARAMETERS =====
CSI_CONFIG = {
    # WiFi parameters
    'frequency': 2.4e9,  # 2.4 GHz
    'wavelength': 0.125,  # meters at 2.4 GHz
    'bandwidth': 20e6,  # 20 MHz
    'num_subcarriers': 30,  # Standard 802.11n
    
    # Antenna configuration (ESP32)
    'antenna_pairs': (2, 2),  # (TX, RX)
    'antenna_spacing': 0.05,  # 5 cm
    'tx_power': 20,  # dBm
    
    # Temporal sampling
    'temporal_samples': 150,  # Samples per packet
    'sampling_rate': 100,  # Hz
    
    # Signal processing
    'phase_filter_size': 5,
    'phase_unwrap': True,
    'amplitude_threshold': -100.0,  # dB
}


# ===== POSTURE DETECTION MODEL =====
POSTURE_MODEL_CONFIG = {
    # Model architecture
    'num_classes': 7,
    'classes': ['standing', 'sitting', 'lying_down', 'walking', 'running', 'bending', 'arm_raising'],
    'cnn_filters': [16, 32],
    'dense_dims': [256, 128],
    'dropout_rate': 0.5,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',
    
    # Data parameters
    'test_size': 0.2,
    'val_size': 0.1,
    'normalize': True,
    'normalize_method': 'z_score',
    
    # Loss function
    'loss_function': 'CrossEntropyLoss',
    'use_class_weights': True,
    
    # LR scheduler
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
}


# ===== PRESENCE DETECTION MODEL =====
PRESENCE_MODEL_CONFIG = {
    # Model type and parameters
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    
    # Training parameters
    'test_size': 0.2,
    'cv_folds': 5,
    'use_grid_search': True,
    'random_state': 42,
    
    # Feature engineering
    'expand_features': True,
    'include_squared_features': True,
    'include_interaction_features': True,
    
    # Scaling
    'scaler': 'StandardScaler',
    'normalize': True,
    
    # Hyperparameter grid for GridSearchCV
    'param_grid': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 3]
    }
}


# ===== 3D KEYPOINT REGRESSION =====
KEYPOINT_MODEL_CONFIG = {
    # Model architecture
    'input_size': 50,
    'hidden_dims': [256, 128, 64],
    'output_size': 3,  # x, y, z
    'dropout_rate': 0.3,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'optimizer': 'Adam',
    'loss_function': 'MSELoss',
    
    # Data parameters
    'test_size': 0.2,
    'normalize': True,
}


# ===== FEATURE EXTRACTION =====
FEATURE_EXTRACTION_CONFIG = {
    'wavelength': 0.125,
    'antenna_spacing': 0.05,
    'ref_rssi': -40.0,
    'path_loss_exponent': 2.0,
    
    # Features to extract
    'extract_amplitude_stats': True,
    'extract_phase_stats': True,
    'extract_distance': True,
    'extract_aoa': True,
    'extract_3d_keypoints': True,
}


# ===== PREPROCESSING =====
PREPROCESSING_CONFIG = {
    'normalize_amplitude': True,
    'normalize_phase': True,
    'normalization_method': 'z_score',
    'remove_outliers': True,
    'outlier_threshold': 3.0,  # Std dev
    'samples_per_group': 150,
}


# ===== OUTPUT PATHS =====
OUTPUT_CONFIG = {
    'checkpoints_dir': './checkpoints',
    'models_dir': './models',
    'results_dir': './results',
    'logs_dir': './logs',
    'plots_dir': './plots',
}


# ===== LOGGING =====
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}


def get_config_dict() -> Dict:
    """
    Get complete configuration dictionary.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        'csi': CSI_CONFIG,
        'posture_model': POSTURE_MODEL_CONFIG,
        'presence_model': PRESENCE_MODEL_CONFIG,
        'keypoint_model': KEYPOINT_MODEL_CONFIG,
        'feature_extraction': FEATURE_EXTRACTION_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'output': OUTPUT_CONFIG,
        'logging': LOGGING_CONFIG,
    }


def print_config():
    """Print all configuration parameters in formatted table."""
    config = get_config_dict()
    
    for category, params in config.items():
        print(f"\n{'='*60}")
        print(f"  {category.upper()}")
        print(f"{'='*60}")
        
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    print_config()
