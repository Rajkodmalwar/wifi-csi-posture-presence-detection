"""
Inference Service Module

Manages pretrained model loading and inference pipeline.
Handles posture detection and presence detection in a single unified interface.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle

from src.preprocessing import CSIPreprocessor, FeatureExtractor
from src.model import PresenceDetectionModel
from src.config import POSTURE_MODEL_CONFIG, PRESENCE_MODEL_CONFIG, CSI_CONFIG

# Try to import PyTorch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Unified inference service for posture and presence detection.
    
    Handles:
    - CSI preprocessing
    - Feature extraction
    - Model inference (both tasks)
    - Result compilation
    """
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize inference service.
        
        Args:
            model_dir: Directory containing pretrained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.csi_preprocessor = CSIPreprocessor(
            subcarriers=CSI_CONFIG['num_subcarriers'],
            antenna_pairs=CSI_CONFIG['antenna_pairs'],
            temporal_samples=CSI_CONFIG['temporal_samples'],
            phase_filter_size=CSI_CONFIG['phase_filter_size']
        )
        
        self.feature_extractor = FeatureExtractor(
            wavelength=CSI_CONFIG['wavelength'],
            antenna_spacing=CSI_CONFIG['antenna_spacing']
        )
        
        # Load pretrained models
        self.posture_model = self._load_posture_model()
        self.presence_model = self._load_presence_model()
        self.posture_label_encoder = self._load_posture_labels()
    
    def _load_posture_model(self):
        """Load pretrained posture detection model."""
        model_path_pth = self.model_dir / "posture_model.pth"
        model_path_pkl = self.model_dir / "posture_model.pkl"
        
        # Try PyTorch model first
        if model_path_pth.exists() and TORCH_AVAILABLE:
            try:
                model = torch.load(model_path_pth, map_location='cpu')
                model.eval()
                logger.info(f"Loaded posture model from {model_path_pth}")
                return model
            except Exception as e:
                logger.error(f"Failed to load PyTorch posture model: {e}")
        
        # Fall back to sklearn model
        if model_path_pkl.exists():
            try:
                with open(model_path_pkl, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded posture model from {model_path_pkl}")
                return model
            except Exception as e:
                logger.error(f"Failed to load sklearn posture model: {e}")
        
        logger.warning("Posture model not found")
        return None
    
    def _load_presence_model(self):
        """Load pretrained presence detection model."""
        model_path = self.model_dir / "presence_model.pkl"
        
        if not model_path.exists():
            logger.warning(f"Presence model not found at {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded presence model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load presence model: {e}")
            return None
    
    def _load_posture_labels(self):
        """Load posture class label encoder."""
        encoder_path = self.model_dir / "posture_label_encoder.pkl"
        
        if not encoder_path.exists():
            logger.warning(f"Label encoder not found at {encoder_path}")
            return None
        
        try:
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            logger.info(f"Loaded label encoder from {encoder_path}")
            return encoder
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
            return None
    
    def preprocess_posture_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess posture CSI data for inference.
        
        Args:
            df: DataFrame with amplitude and phase columns
            
        Returns:
            Tuple of:
            - preprocessed_tensors: [batch, temporal, antenna_pairs]
            - features: [batch, num_features]
            - metadata: preprocessing details for visualization
        """
        try:
            # Extract amplitude and phase columns
            amplitude_cols = [c for c in df.columns if 'amplitude' in c.lower()]
            phase_cols = [c for c in df.columns if 'phase' in c.lower()]
            
            if not amplitude_cols or not phase_cols:
                raise ValueError("CSV must contain 'amplitude' and 'phase' columns")
            
            amplitude_data = df[amplitude_cols].values
            phase_data = df[phase_cols].values
            
            # Normalize
            amplitude_normalized = self._normalize_data(amplitude_data)
            phase_normalized = self._normalize_data(phase_data)
            
            # Reshape to temporal tensors if needed
            batch_size = len(df)
            temporal_size = CSI_CONFIG['temporal_samples']
            antenna_pairs = len(amplitude_cols)  # Use actual number of subcarriers
            
            # Simple reshape - group sequential samples
            # amplitude_normalized: [batch_size, amplitude_cols]
            # Reshape to [batch_size, 1, num_subcarriers] for compatibility
            
            amplitude_tensor = amplitude_normalized.reshape(batch_size, 1, antenna_pairs)
            phase_tensor = phase_normalized.reshape(batch_size, 1, antenna_pairs)
            
            # Extract features
            features_list = []
            for i in range(batch_size):
                amp_feat, phase_feat = self.feature_extractor.extract_amplitude_phase(
                    amplitude_tensor[i],
                    phase_tensor[i]
                )
                features_list.append(np.concatenate([amp_feat, phase_feat]))
            
            features = np.array(features_list)
            
            # Pad features to 24 dimensions if needed
            if features.shape[1] < 24:
                features = np.pad(features, ((0, 0), (0, 24 - features.shape[1])), mode='constant', constant_values=0)
            elif features.shape[1] > 24:
                features = features[:, :24]
            
            metadata = {
                "batch_size": batch_size,
                "num_subcarriers": len(amplitude_cols),
                "amplitude_stats": {
                    "mean": float(amplitude_data.mean()),
                    "std": float(amplitude_data.std()),
                    "min": float(amplitude_data.min()),
                    "max": float(amplitude_data.max())
                },
                "phase_stats": {
                    "mean": float(phase_data.mean()),
                    "std": float(phase_data.std()),
                    "min": float(phase_data.min()),
                    "max": float(phase_data.max())
                }
            }
            
            return amplitude_tensor, phase_tensor, features, metadata
        
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def preprocess_presence_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess presence detection data for inference.
        
        Args:
            df: DataFrame with WiFi parameters (rssi, rate, noise_floor, channel)
            
        Returns:
            Tuple of:
            - features: [batch, num_features]
            - metadata: preprocessing details for visualization
        """
        try:
            required_cols = ['rssi', 'rate', 'noise_floor', 'channel']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            X = df[required_cols].values
            
            # Pad features to 24 dimensions for model compatibility
            if X.shape[1] < 24:
                X = np.pad(X, ((0, 0), (0, 24 - X.shape[1])), mode='constant', constant_values=0)
            
            metadata = {
                "batch_size": len(df),
                "num_features": 4,  # Original feature count
                "column_stats": {}
            }
            
            for col in required_cols:
                metadata["column_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
            
            return X, metadata
        
        except Exception as e:
            logger.error(f"Presence preprocessing error: {e}")
            raise
    
    def infer_posture(
        self,
        features: np.ndarray
    ) -> Dict:
        """
        Infer posture class from features.
        
        Args:
            features: [batch, num_features] feature array
            
        Returns:
            Dict with predictions and confidence scores
        """
        if self.posture_model is None:
            return {
                "error": "Posture model not loaded",
                "predictions": None
            }
        
        try:
            # Handle both PyTorch and sklearn models
            if TORCH_AVAILABLE and hasattr(self.posture_model, 'forward'):
                # PyTorch model
                features_tensor = torch.FloatTensor(features)
                
                with torch.no_grad():
                    outputs = self.posture_model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(probs, dim=1)
                
                predictions = []
                for pred_idx, prob in zip(predicted_classes, probs):
                    pred_label = self.posture_label_encoder.inverse_transform([pred_idx.item()])[0] \
                        if self.posture_label_encoder else f"class_{pred_idx.item()}"
                    
                    predictions.append({
                        "posture": pred_label,
                        "confidence": float(prob.max().item()),
                        "all_scores": {
                            POSTURE_MODEL_CONFIG['classes'][i]: float(prob[i].item())
                            for i in range(len(POSTURE_MODEL_CONFIG['classes']))
                        }
                    })
            else:
                # Sklearn model
                pred_classes = self.posture_model.predict(features)
                pred_probs = self.posture_model.predict_proba(features)
                
                predictions = []
                for pred_idx, probs in zip(pred_classes, pred_probs):
                    # Handle both string labels and integer indices
                    if isinstance(pred_idx, str):
                        pred_label = pred_idx
                    else:
                        # Try to use label encoder if available
                        try:
                            pred_label = self.posture_label_encoder.inverse_transform([pred_idx])[0] \
                                if self.posture_label_encoder else f"class_{pred_idx}"
                        except:
                            pred_label = f"class_{pred_idx}"
                    
                    predictions.append({
                        "posture": pred_label,
                        "confidence": float(probs.max()),
                        "all_scores": {
                            POSTURE_MODEL_CONFIG['classes'][i]: float(probs[i]) if i < len(probs) else 0.0
                            for i in range(len(POSTURE_MODEL_CONFIG['classes']))
                        }
                    })
            
            return {
                "predictions": predictions,
                "num_samples": len(features)
            }
        
        except Exception as e:
            logger.error(f"Posture inference error: {e}")
            return {"error": str(e)}
    
    def infer_presence(
        self,
        features: np.ndarray
    ) -> Dict:
        """
        Infer presence detection from features.
        
        Args:
            features: [batch, num_features] feature array
            
        Returns:
            Dict with presence predictions and confidence scores
        """
        if self.presence_model is None:
            return {
                "error": "Presence model not loaded",
                "predictions": None
            }
        
        try:
            # Get predictions and probabilities
            predictions = self.presence_model.predict(features)
            probabilities = self.presence_model.predict_proba(features)
            
            results = []
            for pred, prob in zip(predictions, probabilities):
                # Determine label based on prediction
                label = "present" if pred == 1 else "absent"
                confidence = float(prob.max())
                
                results.append({
                    "presence": label,
                    "confidence": confidence,
                    "absent_score": float(prob[0]),
                    "present_score": float(prob[1]) if len(prob) > 1 else 0.0
                })
            
            return {
                "predictions": results,
                "num_samples": len(features)
            }
        
        except Exception as e:
            logger.error(f"Presence inference error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization.
        
        Args:
            data: Input array
            
        Returns:
            Normalized array
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (data - mean) / std
