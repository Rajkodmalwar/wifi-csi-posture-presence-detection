"""
Utility Functions Module

General utility functions for logging, data visualization, and system setup.

Functions:
    - setup_logging: Configure logging system
    - create_directories: Create required output directories
    - plot_tensor_sample: Visualize CSI tensor data
    - plot_data_distribution: Plot class distribution
    - compute_class_weights: Calculate weights for imbalanced data
"""

import logging
import logging.handlers
from pathlib import Path
from typing import List, Dict
import numpy as np
from config import LOGGING_CONFIG, OUTPUT_CONFIG
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """
    Setup logging system with file and console handlers.
    
    Args:
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['date_format']
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(output_dir: str = './output'):
    """
    Create required output directories.
    
    Args:
        output_dir: Base output directory path
    """
    output_path = Path(output_dir)
    
    # Create subdirectories
    subdirs = [
        'checkpoints',
        'models',
        'results',
        'logs',
        'plots',
        'data'
    ]
    
    for subdir in subdirs:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created output directories in {output_path}")


def plot_tensor_sample(
    amplitude_tensor: np.ndarray,
    phase_tensor: np.ndarray,
    title: str = 'CSI Tensor Sample',
    save_path: str = None
):
    """
    Plot amplitude and phase tensors side by side.
    
    Args:
        amplitude_tensor: [temporal, antenna_pairs] array
        phase_tensor: [temporal, antenna_pairs] array
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Amplitude plot
    sns.heatmap(amplitude_tensor.T, cmap='viridis', ax=axes[0], cbar_kws={'label': 'Amplitude (dB)'})
    axes[0].set_title('Amplitude')
    axes[0].set_xlabel('Temporal Sample')
    axes[0].set_ylabel('Antenna Pair')
    
    # Phase plot
    sns.heatmap(phase_tensor.T, cmap='hsv', ax=axes[1], cbar_kws={'label': 'Phase (radians)'})
    axes[1].set_title('Phase')
    axes[1].set_xlabel('Temporal Sample')
    axes[1].set_ylabel('Antenna Pair')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def plot_data_distribution(
    labels: np.ndarray,
    class_names: List[str] = None,
    title: str = 'Data Distribution',
    save_path: str = None
):
    """
    Plot class distribution bar chart.
    
    Args:
        labels: Array of class labels
        class_names: Optional list of class names
        title: Plot title
        save_path: Optional path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color='steelblue')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced data.
    
    Weights = num_samples / (num_classes * samples_per_class)
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(unique)
    
    weights = {}
    for class_idx, count in zip(unique, counts):
        weight = total_samples / (num_classes * count)
        weights[class_idx] = weight
    
    return weights


def plot_feature_statistics(
    features: np.ndarray,
    feature_names: List[str] = None,
    save_path: str = None
):
    """
    Plot feature statistics (mean, std, min, max).
    
    Args:
        features: [num_samples, num_features] array
        feature_names: Optional list of feature names
        save_path: Optional path to save figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(features.shape[1])]
    
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    mins = np.min(features, axis=0)
    maxs = np.max(features, axis=0)
    
    x = np.arange(len(feature_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width*1.5, means, width, label='Mean', alpha=0.8)
    ax.bar(x - width/2, stds, width, label='Std', alpha=0.8)
    ax.bar(x + width/2, mins, width, label='Min', alpha=0.8)
    ax.bar(x + width*1.5, maxs, width, label='Max', alpha=0.8)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.set_title('Feature Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
