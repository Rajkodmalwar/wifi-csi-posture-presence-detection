"""
CSI Data Preprocessing Module

This module handles the complete preprocessing pipeline for WiFi CSI data:
- Loading raw CSI from CSV files
- Phase unwrapping and filtering
- Amplitude and phase normalization
- Tensor creation for model input

Classes:
    CSIPreprocessor: Main preprocessing class with all data transformation methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Optional
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


class CSIPreprocessor:
    """
    Preprocesses raw WiFi CSI CSV data into normalized amplitude and phase tensors.
    
    The preprocessing pipeline includes:
    1. Parse raw CSI values from CSV
    2. Unwrap phase discontinuities
    3. Normalize amplitude and phase using Z-score normalization
    4. Create 3D tensors: [batch, temporal, antennas]
    
    Attributes:
        subcarriers (int): Number of WiFi subcarriers (default: 30)
        antenna_pairs (Tuple[int, int]): Antenna configuration (default: 2x2)
        temporal_samples (int): Number of temporal samples (default: 150)
        phase_filter_size (int): Median filter window for phase (default: 5)
    """
    
    def __init__(
        self,
        subcarriers: int = 30,
        antenna_pairs: Tuple[int, int] = (2, 2),
        temporal_samples: int = 150,
        phase_filter_size: int = 5,
        amplitude_threshold: float = -100.0
    ):
        """
        Initialize CSI preprocessor with configuration parameters.
        
        Args:
            subcarriers: Number of WiFi subcarriers
            antenna_pairs: Antenna configuration (tx, rx)
            temporal_samples: Number of temporal samples per packet
            phase_filter_size: Window size for median filtering
            amplitude_threshold: Minimum amplitude threshold (dB)
        """
        self.subcarriers = subcarriers
        self.antenna_pairs = antenna_pairs
        self.temporal_samples = temporal_samples
        self.phase_filter_size = phase_filter_size
        self.amplitude_threshold = amplitude_threshold
        
    def preprocess_csv(
        self,
        csv_path: str,
        samples_per_group: int = 150,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CSI data from CSV file.
        
        Args:
            csv_path: Path to CSV file with raw CSI data
            samples_per_group: Number of samples to group into tensor
            normalize: Whether to apply Z-score normalization
            
        Returns:
            Tuple of (amplitude_tensors, phase_tensors, labels)
            - amplitude_tensors: [batch, temporal, antenna_pairs]
            - phase_tensors: [batch, temporal, antenna_pairs]
            - labels: [batch] class labels
        """
        logger.info(f"Loading CSI data from {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract amplitude and phase
        amplitude_list = []
        phase_list = []
        labels = []
        
        # Group data by label if available
        if 'label' in df.columns:
            grouped = df.groupby('label')
        else:
            grouped = [(None, df)]
        
        for label, group_df in grouped:
            # Convert amplitude and phase columns to arrays
            amp_data = self._extract_csi_values(group_df, 'amplitude')
            phase_data = self._extract_csi_values(group_df, 'phase')
            
            # Unwrap and filter phase
            phase_data = self.unwrap_phase(phase_data)
            phase_data = self.sanitize_phase(phase_data)
            
            # Create tensors
            amp_tensors = self._create_tensors(amp_data, samples_per_group)
            phase_tensors = self._create_tensors(phase_data, samples_per_group)
            
            amplitude_list.extend(amp_tensors)
            phase_list.extend(phase_tensors)
            
            if label is not None:
                labels.extend([label] * len(amp_tensors))
        
        # Convert to numpy arrays
        amplitude_tensors = np.array(amplitude_list)
        phase_tensors = np.array(phase_list)
        labels = np.array(labels) if labels else np.zeros(len(amplitude_tensors))
        
        # Normalize
        if normalize:
            amplitude_tensors = self._normalize_z_score(amplitude_tensors)
            phase_tensors = self._normalize_z_score(phase_tensors)
        
        logger.info(f"Created {len(amplitude_tensors)} tensors")
        logger.info(f"Amplitude shape: {amplitude_tensors.shape}, Phase shape: {phase_tensors.shape}")
        
        return amplitude_tensors, phase_tensors, labels
    
    def _extract_csi_values(self, df: pd.DataFrame, value_type: str) -> np.ndarray:
        """
        Extract CSI amplitude or phase values from DataFrame.
        
        Args:
            df: DataFrame with CSI data
            value_type: 'amplitude' or 'phase'
            
        Returns:
            Array of extracted values
        """
        columns = [col for col in df.columns if value_type in col.lower()]
        if not columns:
            raise ValueError(f"No {value_type} columns found in CSV")
        
        return df[columns].values
    
    def unwrap_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """
        Unwrap phase discontinuities using NumPy's unwrap function.
        
        Removes 2π jumps in phase data to create continuous signal.
        
        Args:
            phase_data: Raw phase data array
            
        Returns:
            Unwrapped phase data
        """
        if phase_data.ndim == 1:
            return np.unwrap(phase_data)
        
        # For multi-dimensional arrays, unwrap along first axis
        unwrapped = np.zeros_like(phase_data, dtype=float)
        for i in range(phase_data.shape[1:]):
            if phase_data.ndim == 2:
                unwrapped[:, i] = np.unwrap(phase_data[:, i])
            else:
                unwrapped[:, i, :] = np.unwrap(phase_data[:, i, :], axis=0)
        
        return unwrapped
    
    def sanitize_phase(self, phase_data: np.ndarray, filter_size: int = 5) -> np.ndarray:
        """
        Sanitize phase data with median filtering and linear fitting.
        
        Applies median filter to smooth noise and linear fitting to remove trends.
        
        Args:
            phase_data: Unwrapped phase data
            filter_size: Median filter window size
            
        Returns:
            Sanitized phase data
        """
        # Apply median filter
        if phase_data.ndim == 1:
            filtered = median_filter(phase_data, size=filter_size)
        else:
            filtered = np.zeros_like(phase_data)
            for i in range(phase_data.shape[1]):
                filtered[:, i] = median_filter(phase_data[:, i], size=filter_size)
        
        return filtered
    
    def _create_tensors(self, data: np.ndarray, samples_per_group: int) -> List[np.ndarray]:
        """
        Create tensors by grouping samples.
        
        Args:
            data: Input data array
            samples_per_group: Samples to group together
            
        Returns:
            List of tensors
        """
        tensors = []
        for i in range(0, len(data) - samples_per_group + 1, samples_per_group):
            tensor = data[i:i + samples_per_group]
            tensors.append(tensor)
        
        return tensors
    
    def _normalize_z_score(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization to data.
        
        Normalizes across each tensor to zero mean and unit variance.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data
        """
        mean = np.mean(data, axis=tuple(range(1, data.ndim)), keepdims=True)
        std = np.std(data, axis=tuple(range(1, data.ndim)), keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        
        return (data - mean) / std
    
    def save_preprocessed_data(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        labels: np.ndarray,
        output_dir: str
    ):
        """
        Save preprocessed tensors to NumPy files.
        
        Args:
            amplitude: Amplitude tensor array
            phase: Phase tensor array
            labels: Label array
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'amplitude_tensors.npy', amplitude)
        np.save(output_dir / 'phase_tensors.npy', phase)
        np.save(output_dir / 'labels.npy', labels)
        
        logger.info(f"Saved preprocessed data to {output_dir}")
