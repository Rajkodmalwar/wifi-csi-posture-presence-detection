"""
Feature Extraction from CSI Data Module

Extracts meaningful features from WiFi CSI tensors:
- Distance estimation using path loss
- Angle of Arrival (AoA) using phase differences
- 3D Cartesian coordinates (x, y, z)
- Amplitude and phase statistics

Classes:
    FeatureExtractor: Main feature extraction class with all feature computation methods
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract physical features from WiFi CSI amplitude and phase tensors.
    
    Uses path loss model for distance estimation and phase differences for
    angle of arrival estimation. Converts to 3D Cartesian coordinates.
    
    Attributes:
        wavelength (float): WiFi wavelength at 2.4 GHz (meters)
        antenna_spacing (float): Distance between antenna elements (meters)
        ref_rssi (float): Reference RSSI at 1 meter (dB)
        path_loss_exponent (float): Path loss exponent (typically 2-4)
    """
    
    def __init__(
        self,
        wavelength: float = 0.125,  # 2.4 GHz: ~12.5 cm
        antenna_spacing: float = 0.05,  # 5 cm typical
        ref_rssi: float = -40.0,  # Reference RSSI
        path_loss_exponent: float = 2.0
    ):
        """
        Initialize feature extractor with WiFi parameters.
        
        Args:
            wavelength: WiFi wavelength in meters
            antenna_spacing: Antenna element spacing in meters
            ref_rssi: Reference RSSI at 1 meter
            path_loss_exponent: Path loss exponent for distance model
        """
        self.wavelength = wavelength
        self.antenna_spacing = antenna_spacing
        self.ref_rssi = ref_rssi
        self.path_loss_exponent = path_loss_exponent
    
    def extract_amplitude_phase(
        self,
        amplitude_tensor: np.ndarray,
        phase_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitude and phase statistics from tensors.
        
        Computes mean, std, min, max for each tensor.
        
        Args:
            amplitude_tensor: [temporal, antenna_pairs] shaped tensor
            phase_tensor: [temporal, antenna_pairs] shaped tensor
            
        Returns:
            Tuple of (amplitude_features, phase_features)
            - Each: [8] shaped (mean, std, min, max per antenna pair)
        """
        amp_mean = np.mean(amplitude_tensor, axis=0)
        amp_std = np.std(amplitude_tensor, axis=0)
        amp_min = np.min(amplitude_tensor, axis=0)
        amp_max = np.max(amplitude_tensor, axis=0)
        
        phase_mean = np.mean(phase_tensor, axis=0)
        phase_std = np.std(phase_tensor, axis=0)
        phase_min = np.min(phase_tensor, axis=0)
        phase_max = np.max(phase_tensor, axis=0)
        
        amp_features = np.concatenate([amp_mean.flatten(), amp_std.flatten(),
                                       amp_min.flatten(), amp_max.flatten()])
        phase_features = np.concatenate([phase_mean.flatten(), phase_std.flatten(),
                                         phase_min.flatten(), phase_max.flatten()])
        
        return amp_features, phase_features
    
    def compute_distance(self, amplitude_rssi: np.ndarray) -> float:
        """
        Estimate distance from device using path loss model.
        
        Distance = 10^((RSSI_ref - RSSI) / (10 * n))
        where n is path loss exponent
        
        Args:
            amplitude_rssi: Mean amplitude as RSSI (dB)
            
        Returns:
            Estimated distance in meters
        """
        if amplitude_rssi >= self.ref_rssi:
            return 1.0  # Very close, return minimum distance
        
        # Path loss formula
        distance = 10 ** ((self.ref_rssi - amplitude_rssi) / 
                         (10 * self.path_loss_exponent))
        
        return np.clip(distance, 0.1, 50.0)  # Realistic range
    
    def compute_aoa(self, phase_tensor: np.ndarray) -> Tuple[float, float]:
        """
        Estimate Angle of Arrival (AoA) using phase differences.
        
        Uses phase difference between antenna elements to compute azimuth
        and elevation angles.
        
        Args:
            phase_tensor: [temporal, 2, 2] shaped phase tensor
            
        Returns:
            Tuple of (azimuth_angle, elevation_angle) in degrees
        """
        # Compute mean phase across temporal dimension
        mean_phase = np.mean(phase_tensor, axis=0)  # [2, 2]
        
        # Phase difference between TX antennas
        phase_diff_tx = mean_phase[0, 0] - mean_phase[1, 0]
        
        # Phase difference between RX antennas  
        phase_diff_rx = mean_phase[0, 0] - mean_phase[0, 1]
        
        # Compute angles using phase differences
        # AoA = arcsin((phase_diff * wavelength) / (2 * pi * antenna_spacing))
        aoa_rad = np.arcsin(np.clip(phase_diff_tx / (2 * np.pi * self.wavelength / 
                                     self.antenna_spacing), -1, 1))
        azimuth = np.degrees(aoa_rad)
        
        eoa_rad = np.arcsin(np.clip(phase_diff_rx / (2 * np.pi * self.wavelength / 
                                     self.antenna_spacing), -1, 1))
        elevation = np.degrees(eoa_rad)
        
        return azimuth, elevation
    
    def compute_3d_keypoints(
        self,
        amplitude_rssi: float,
        azimuth_angle: float,
        elevation_angle: float
    ) -> Tuple[float, float, float]:
        """
        Convert distance and angles to 3D Cartesian coordinates.
        
        Args:
            amplitude_rssi: RSSI amplitude (dB)
            azimuth_angle: Azimuth angle in degrees
            elevation_angle: Elevation angle in degrees
            
        Returns:
            Tuple of (x, y, z) 3D coordinates in meters
        """
        distance = self.compute_distance(amplitude_rssi)
        
        # Convert angles to radians
        az_rad = np.radians(azimuth_angle)
        el_rad = np.radians(elevation_angle)
        
        # Spherical to Cartesian conversion
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)
        
        return float(x), float(y), float(z)
    
    def extract_all_features(
        self,
        amplitude_tensor: np.ndarray,
        phase_tensor: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract all features from a CSI tensor pair.
        
        Combines amplitude, phase, distance, AoA, and 3D keypoint features.
        
        Args:
            amplitude_tensor: [temporal, antenna_pairs] shaped tensor
            phase_tensor: [temporal, antenna_pairs] shaped tensor
            
        Returns:
            Dictionary with feature names and values
        """
        # Extract amplitude and phase stats
        amp_features, phase_features = self.extract_amplitude_phase(
            amplitude_tensor, phase_tensor
        )
        
        # Compute distance
        amplitude_rssi = np.mean(amplitude_tensor)
        distance = self.compute_distance(amplitude_rssi)
        
        # Compute angles
        azimuth, elevation = self.compute_aoa(phase_tensor)
        
        # Compute 3D keypoints
        x, y, z = self.compute_3d_keypoints(amplitude_rssi, azimuth, elevation)
        
        # Assemble feature dictionary
        features = {
            'amplitude_mean': amp_features[0],
            'amplitude_std': amp_features[1],
            'amplitude_min': amp_features[2],
            'amplitude_max': amp_features[3],
            'phase_mean': phase_features[0],
            'phase_std': phase_features[1],
            'phase_min': phase_features[2],
            'phase_max': phase_features[3],
            'distance': distance,
            'azimuth': azimuth,
            'elevation': elevation,
            'keypoint_x': x,
            'keypoint_y': y,
            'keypoint_z': z,
            'rssi': amplitude_rssi
        }
        
        return features
    
    def extract_batch_features(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Extract features from a batch of tensors.
        
        Args:
            amplitude_tensors: [batch, temporal, antenna_pairs] array
            phase_tensors: [batch, temporal, antenna_pairs] array
            labels: Optional [batch] label array
            
        Returns:
            DataFrame with features as columns and samples as rows
        """
        features_list = []
        
        for i in range(len(amplitude_tensors)):
            features = self.extract_all_features(
                amplitude_tensors[i],
                phase_tensors[i]
            )
            
            if labels is not None:
                features['label'] = labels[i]
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
