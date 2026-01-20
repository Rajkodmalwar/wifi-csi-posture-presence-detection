"""
Model Package for WiFi CSI Detection

Contains all deep learning and machine learning models:
- train.py: Posture detection CNN training
- test.py: Model evaluation and inference
- keypoint_regression.py: 3D body keypoint prediction
- presence_detection.py: Random Forest presence/absence classifier
"""

from .train import CSIModalityNetwork, CSIDataset, CSIModelTrainer
from .test import CSIModelEvaluator
from .keypoint_regression import KeypointNet, KeypointTrainer
from .presence_detection import PresenceDetectionModel

__all__ = [
    'CSIModalityNetwork', 'CSIDataset', 'CSIModelTrainer',
    'CSIModelEvaluator',
    'KeypointNet', 'KeypointTrainer',
    'PresenceDetectionModel'
]
