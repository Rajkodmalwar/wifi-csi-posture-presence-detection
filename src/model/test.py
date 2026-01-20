"""
Model Evaluation and Testing Module

Provides evaluation metrics, predictions, and visualization for posture detection model.

Classes:
    CSIModelEvaluator: Comprehensive model evaluation with metrics and plots
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

logger = logging.getLogger(__name__)


class CSIModelEvaluator:
    """
    Comprehensive model evaluation toolkit.
    
    Computes accuracy, precision, recall, F1, confusion matrix,
    and per-class performance metrics. Includes visualization functions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        label_encoder: object,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model instance
            label_encoder: LabelEncoder for class conversion
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.label_encoder = label_encoder
        self.device = torch.device(device)
        self.model.to(self.device)
    
    def predict_batch(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on batch of samples.
        
        Args:
            amplitude_tensors: [batch, temporal, antenna_pairs] array
            phase_tensors: [batch, temporal, antenna_pairs] array
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.model.eval()
        
        amplitude = torch.FloatTensor(amplitude_tensors)
        phase = torch.FloatTensor(phase_tensors)
        
        # Add channel dimension if needed
        if amplitude.dim() == 2:
            amplitude = amplitude.unsqueeze(1)
        if phase.dim() == 2:
            phase = phase.unsqueeze(1)
        
        amplitude = amplitude.to(self.device)
        phase = phase.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(amplitude, phase)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy
        pred_classes = predictions.cpu().numpy()
        pred_probs = probabilities.cpu().numpy()
        
        # Decode labels
        decoded_classes = self.label_encoder.inverse_transform(pred_classes)
        
        return decoded_classes, pred_probs
    
    def evaluate(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            amplitude_tensors: Test amplitude tensors
            phase_tensors: Test phase tensors
            true_labels: Ground truth labels
            
        Returns:
            Dictionary with metrics
        """
        # Get predictions
        pred_labels, _ = self.predict_batch(amplitude_tensors, phase_tensors)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'f1': f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class performance metrics.
        
        Args:
            amplitude_tensors: Test amplitude tensors
            phase_tensors: Test phase tensors
            true_labels: Ground truth labels
            
        Returns:
            Dictionary with per-class metrics
        """
        pred_labels, _ = self.predict_batch(amplitude_tensors, phase_tensors)
        
        # Get unique classes
        classes = self.label_encoder.classes_
        
        per_class_metrics = {}
        for cls in classes:
            mask = true_labels == cls
            if mask.sum() == 0:
                continue
            
            pred_cls = pred_labels == cls
            true_cls = true_labels == cls
            
            tp = (pred_cls & true_cls).sum()
            fp = (pred_cls & ~true_cls).sum()
            fn = (~pred_cls & true_cls).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': mask.sum()
            }
        
        return per_class_metrics
    
    def plot_confusion_matrix(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        true_labels: np.ndarray,
        save_path: str = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            amplitude_tensors: Test amplitude tensors
            phase_tensors: Test phase tensors
            true_labels: Ground truth labels
            save_path: Optional path to save figure
        """
        pred_labels, _ = self.predict_batch(amplitude_tensors, phase_tensors)
        
        cm = confusion_matrix(true_labels, pred_labels,
                             labels=self.label_encoder.classes_)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def plot_per_class_performance(
        self,
        per_class_metrics: Dict,
        save_path: str = None
    ):
        """
        Plot per-class performance metrics.
        
        Args:
            per_class_metrics: Dictionary from compute_per_class_metrics
            save_path: Optional path to save figure
        """
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[c]['precision'] for c in classes]
        recall = [per_class_metrics[c]['recall'] for c in classes]
        f1 = [per_class_metrics[c]['f1'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Per-class metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        true_labels: np.ndarray
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            amplitude_tensors: Test amplitude tensors
            phase_tensors: Test phase tensors
            true_labels: Ground truth labels
            
        Returns:
            Formatted report string
        """
        pred_labels, _ = self.predict_batch(amplitude_tensors, phase_tensors)
        
        report = classification_report(
            true_labels, pred_labels,
            target_names=self.label_encoder.classes_
        )
        
        return report
