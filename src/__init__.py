"""
WiFi CSI Detection Package - Main Module

Complete package for WiFi-based posture and presence detection using CSI data.

Features:
    - CSI preprocessing and normalization
    - Feature extraction (distance, AoA, 3D keypoints)
    - Deep learning model for posture classification
    - Random Forest model for presence detection
    - Comprehensive evaluation and visualization

Modules:
    - preprocessing: CSI data loading and feature extraction
    - model: Deep learning and ML models
    - config: Configuration parameters
    - utils: Utility functions
    - data_utils: Data loading and validation
"""

__version__ = '1.0.0'
__author__ = 'CSI Detection Team'

from .preprocessing import CSIPreprocessor, FeatureExtractor
from .model import CSIModalityNetwork, CSIModelTrainer, CSIModelEvaluator
from .model import KeypointNet, KeypointTrainer
from .model import PresenceDetectionModel

__all__ = [
    'CSIPreprocessor',
    'FeatureExtractor',
    'CSIModalityNetwork',
    'CSIModelTrainer',
    'CSIModelEvaluator',
    'KeypointNet',
    'KeypointTrainer',
    'PresenceDetectionModel'
]
