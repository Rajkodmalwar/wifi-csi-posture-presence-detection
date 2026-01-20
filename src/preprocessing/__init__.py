"""
WiFi CSI Preprocessing Package

This package contains all preprocessing and feature extraction utilities
for WiFi CSI (Channel State Information) data.

Modules:
    - csi_preprocessing: Raw CSI CSV data loading and tensor creation
    - feature_extraction: Feature extraction from CSI tensors (distance, AoA, 3D keypoints)
"""

from .csi_preprocessing import CSIPreprocessor
from .feature_extraction import FeatureExtractor

__all__ = ['CSIPreprocessor', 'FeatureExtractor']
