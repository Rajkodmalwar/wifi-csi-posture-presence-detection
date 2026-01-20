"""
Data Utilities Module

Complete data pipeline: loading, validation, splitting, and statistics computation.

Classes:
    - DataLoader: Load CSI data from various formats
    - DataValidator: Check data integrity and distribution
    - DataSplitter: Train/val/test splitting with stratification
    - DataStatistics: Compute statistical properties
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


class DataLoader:
    """Load CSI data from various file formats."""
    
    @staticmethod
    def load_csi_csv(
        csv_path: str,
        amplitude_cols: List[str] = None,
        phase_cols: List[str] = None,
        label_col: str = 'label'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CSI data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            amplitude_cols: Columns for amplitude data
            phase_cols: Columns for phase data
            label_col: Column for labels
            
        Returns:
            Tuple of (amplitude, phase, labels)
        """
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Auto-detect columns if not provided
        if amplitude_cols is None:
            amplitude_cols = [col for col in df.columns if 'amplitude' in col.lower() or 'rssi' in col.lower()]
        if phase_cols is None:
            phase_cols = [col for col in df.columns if 'phase' in col.lower()]
        
        amplitude = df[amplitude_cols].values if amplitude_cols else None
        phase = df[phase_cols].values if phase_cols else None
        labels = df[label_col].values if label_col in df.columns else np.zeros(len(df))
        
        return amplitude, phase, labels
    
    @staticmethod
    def load_numpy_arrays(
        amplitude_path: str,
        phase_path: str,
        label_path: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from NumPy files.
        
        Args:
            amplitude_path: Path to amplitude .npy file
            phase_path: Path to phase .npy file
            label_path: Optional path to labels .npy file
            
        Returns:
            Tuple of (amplitude, phase, labels)
        """
        amplitude = np.load(amplitude_path)
        phase = np.load(phase_path)
        labels = np.load(label_path) if label_path else np.zeros(len(amplitude))
        
        logger.info(f"Loaded amplitude shape: {amplitude.shape}")
        logger.info(f"Loaded phase shape: {phase.shape}")
        
        return amplitude, phase, labels
    
    @staticmethod
    def load_features_csv(
        csv_path: str,
        feature_cols: List[str] = None,
        label_col: str = 'label'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load extracted features from CSV.
        
        Args:
            csv_path: Path to CSV file
            feature_cols: Columns to load as features
            label_col: Column for labels
            
        Returns:
            Tuple of (features, labels)
        """
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != label_col]
        
        features = df[feature_cols].values
        labels = df[label_col].values if label_col in df.columns else np.zeros(len(df))
        
        logger.info(f"Loaded features shape: {features.shape}")
        
        return features, labels


class DataValidator:
    """Validate data integrity and distribution."""
    
    @staticmethod
    def check_missing_values(data: np.ndarray) -> Dict[str, any]:
        """
        Check for missing (NaN/Inf) values.
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with validation results
        """
        has_nan = np.isnan(data).any()
        nan_count = np.isnan(data).sum()
        has_inf = np.isinf(data).any()
        inf_count = np.isinf(data).sum()
        
        return {
            'has_nan': has_nan,
            'nan_count': nan_count,
            'has_inf': has_inf,
            'inf_count': inf_count,
            'valid': not has_nan and not has_inf
        }
    
    @staticmethod
    def check_data_ranges(
        data: np.ndarray,
        min_val: float = -200,
        max_val: float = 200
    ) -> Dict[str, any]:
        """
        Check if data values are within expected range.
        
        Args:
            data: Data array
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Dictionary with validation results
        """
        out_of_range = np.sum((data < min_val) | (data > max_val))
        
        return {
            'min': np.min(data),
            'max': np.max(data),
            'out_of_range_count': out_of_range,
            'out_of_range_ratio': out_of_range / data.size,
            'valid': out_of_range == 0
        }
    
    @staticmethod
    def check_class_balance(labels: np.ndarray) -> Dict[str, any]:
        """
        Check class distribution balance.
        
        Args:
            labels: Class label array
            
        Returns:
            Dictionary with balance statistics
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        balance_ratio = counts.max() / counts.min()
        
        return {
            'classes': unique,
            'counts': counts,
            'balance_ratio': balance_ratio,
            'is_balanced': balance_ratio < 3.0  # Threshold for balance
        }
    
    @staticmethod
    def check_tensor_shape(
        tensor: np.ndarray,
        expected_shape: Tuple = None
    ) -> Dict[str, any]:
        """
        Check tensor shape validity.
        
        Args:
            tensor: Tensor to check
            expected_shape: Expected shape
            
        Returns:
            Dictionary with shape validation results
        """
        matches = True
        if expected_shape is not None:
            matches = tensor.shape == expected_shape
        
        return {
            'shape': tensor.shape,
            'ndim': tensor.ndim,
            'size': tensor.size,
            'matches_expected': matches
        }


class DataSplitter:
    """Split data for training, validation, and testing."""
    
    @staticmethod
    def train_val_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            X: Feature array
            y: Label array
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def kfold_split(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate stratified K-fold splits.
        
        Args:
            X: Feature array
            y: Label array
            n_splits: Number of folds
            random_state: Random seed
            
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, val_idx in skf.split(X, y):
            splits.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
        
        return splits


class DataStatistics:
    """Compute statistical properties of data."""
    
    @staticmethod
    def compute_tensor_statistics(tensors: np.ndarray) -> Dict[str, any]:
        """
        Compute statistics for tensor data.
        
        Args:
            tensors: [batch, ...] tensor array
            
        Returns:
            Dictionary with statistics
        """
        return {
            'shape': tensors.shape,
            'dtype': tensors.dtype,
            'min': np.min(tensors),
            'max': np.max(tensors),
            'mean': np.mean(tensors),
            'std': np.std(tensors),
            'median': np.median(tensors),
            'q25': np.percentile(tensors, 25),
            'q75': np.percentile(tensors, 75)
        }
    
    @staticmethod
    def compute_feature_statistics(
        features: np.ndarray,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute per-feature statistics.
        
        Args:
            features: [num_samples, num_features] array
            feature_names: Optional feature names
            
        Returns:
            DataFrame with per-feature statistics
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
        
        stats = {
            'Feature': feature_names,
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0),
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Median': np.median(features, axis=0),
        }
        
        return pd.DataFrame(stats)
