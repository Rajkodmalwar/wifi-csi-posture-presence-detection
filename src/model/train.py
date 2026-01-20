"""
Posture Detection Model Training Module

Implements CNN architecture with dual modality branches for amplitude and phase.
Includes training loop, validation, loss computation, and model checkpointing.

Classes:
    CSIModalityNetwork: CNN model with separate amplitude/phase branches
    CSIDataset: PyTorch Dataset for CSI tensors
    CSIModelTrainer: Training orchestration and management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

logger = logging.getLogger(__name__)


class CSIModalityNetwork(nn.Module):
    """
    CNN with dual modality branches for CSI amplitude and phase.
    
    Architecture:
    - Amplitude branch: Conv(4→16→32) → Flatten → Dense(512)
    - Phase branch: Conv(4→16→32) → Flatten → Dense(512)
    - Fusion: Concatenate → Dense(256) → Output
    
    Input shape: [batch, temporal=150, antenna_pairs=4]
    Output: [batch, num_classes]
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        cnn_filters: List[int] = None,
        dense_dims: List[int] = None,
        dropout_rate: float = 0.5
    ):
        """
        Initialize CSI modality network.
        
        Args:
            num_classes: Number of output classes
            cnn_filters: CNN filter sizes per layer
            dense_dims: Dense layer dimensions
            dropout_rate: Dropout probability
        """
        super(CSIModalityNetwork, self).__init__()
        
        if cnn_filters is None:
            cnn_filters = [16, 32]
        if dense_dims is None:
            dense_dims = [256, 128]
        
        self.num_classes = num_classes
        
        # ===== Amplitude Branch =====
        self.amp_conv1 = nn.Conv1d(4, cnn_filters[0], kernel_size=3, padding=1)
        self.amp_bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.amp_conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.amp_bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.amp_pool = nn.AdaptiveAvgPool1d(10)
        
        # Calculate flattened size: cnn_filters[1] * 10
        self.amp_flatten_size = cnn_filters[1] * 10
        self.amp_fc = nn.Linear(self.amp_flatten_size, dense_dims[0])
        self.amp_bn3 = nn.BatchNorm1d(dense_dims[0])
        
        # ===== Phase Branch =====
        self.phase_conv1 = nn.Conv1d(4, cnn_filters[0], kernel_size=3, padding=1)
        self.phase_bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.phase_conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.phase_bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.phase_pool = nn.AdaptiveAvgPool1d(10)
        self.phase_fc = nn.Linear(self.amp_flatten_size, dense_dims[0])
        self.phase_bn3 = nn.BatchNorm1d(dense_dims[0])
        
        # ===== Fusion Layers =====
        self.fusion_fc1 = nn.Linear(dense_dims[0] * 2, dense_dims[1])
        self.fusion_bn = nn.BatchNorm1d(dense_dims[1])
        self.fusion_fc2 = nn.Linear(dense_dims[1], num_classes)
        
        # ===== Common =====
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            amplitude: [batch, 4, temporal=150] tensor
            phase: [batch, 4, temporal=150] tensor
            
        Returns:
            [batch, num_classes] output logits
        """
        # ===== Amplitude Branch =====
        amp_x = self.relu(self.amp_bn1(self.amp_conv1(amplitude)))
        amp_x = self.relu(self.amp_bn2(self.amp_conv2(amp_x)))
        amp_x = self.amp_pool(amp_x)
        amp_x = amp_x.view(amp_x.size(0), -1)
        amp_x = self.dropout(amp_x)
        amp_x = self.relu(self.amp_bn3(self.amp_fc(amp_x)))
        amp_x = self.dropout(amp_x)
        
        # ===== Phase Branch =====
        phase_x = self.relu(self.phase_bn1(self.phase_conv1(phase)))
        phase_x = self.relu(self.phase_bn2(self.phase_conv2(phase_x)))
        phase_x = self.phase_pool(phase_x)
        phase_x = phase_x.view(phase_x.size(0), -1)
        phase_x = self.dropout(phase_x)
        phase_x = self.relu(self.phase_bn3(self.phase_fc(phase_x)))
        phase_x = self.dropout(phase_x)
        
        # ===== Fusion =====
        fused = torch.cat([amp_x, phase_x], dim=1)
        fused = self.dropout(fused)
        fused = self.relu(self.fusion_bn(self.fusion_fc1(fused)))
        fused = self.dropout(fused)
        output = self.fusion_fc2(fused)
        
        return output


class CSIDataset(Dataset):
    """
    PyTorch Dataset for CSI amplitude and phase tensors.
    
    Handles loading, reshaping, and serving tensors with labels.
    """
    
    def __init__(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        labels: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            amplitude_tensors: [batch, temporal, antenna_pairs] array
            phase_tensors: [batch, temporal, antenna_pairs] array
            labels: [batch] label array
        """
        self.amplitude = torch.FloatTensor(amplitude_tensors)
        self.phase = torch.FloatTensor(phase_tensors)
        self.labels = torch.LongTensor(labels)
        
        # Reshape to [batch, channels, temporal]
        if self.amplitude.dim() == 2:
            self.amplitude = self.amplitude.unsqueeze(1)
        if self.phase.dim() == 2:
            self.phase = self.phase.unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.amplitude)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Tuple of (amplitude_tensor, phase_tensor, label)
        """
        return self.amplitude[idx], self.phase[idx], self.labels[idx]


class CSIModelTrainer:
    """
    Training manager for CSI posture detection model.
    
    Handles training loop, validation, learning rate scheduling,
    and model checkpointing.
    
    Attributes:
        model: CSIModalityNetwork instance
        device: torch device (cpu or cuda)
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model = CSIModalityNetwork(num_classes=num_classes).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.label_encoder = None
        self.class_weights = None
    
    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        Compute class weights to handle imbalanced data.
        
        Args:
            labels: Class labels
            
        Returns:
            Tensor of class weights
        """
        unique, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(unique) * counts)
        weights = torch.FloatTensor(weights).to(self.device)
        self.class_weights = weights
        return weights
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for amplitude, phase, labels in train_loader:
            amplitude = amplitude.to(self.device)
            phase = phase.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(amplitude, phase)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for amplitude, phase, labels in val_loader:
                amplitude = amplitude.to(self.device)
                phase = phase.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(amplitude, phase)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return total_loss / len(val_loader), accuracy
    
    def fit(
        self,
        amplitude_tensors: np.ndarray,
        phase_tensors: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 8,
        test_size: float = 0.2,
        val_size: float = 0.1,
        save_dir: str = './checkpoints'
    ) -> Dict[str, List[float]]:
        """
        Train model on full dataset.
        
        Args:
            amplitude_tensors: Amplitude CSI tensors
            phase_tensors: Phase CSI tensors
            labels: Class labels
            epochs: Number of epochs
            batch_size: Training batch size
            test_size: Fraction for test set
            val_size: Fraction for validation set
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Compute class weights
        self.compute_class_weights(encoded_labels)
        
        # Split data
        idx = np.arange(len(amplitude_tensors))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Create datasets and loaders
        train_dataset = CSIDataset(
            amplitude_tensors[train_idx],
            phase_tensors[train_idx],
            encoded_labels[train_idx]
        )
        val_dataset = CSIDataset(
            amplitude_tensors[val_idx],
            phase_tensors[val_idx],
            encoded_labels[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / 'posture_model.pth')
        pickle.dump(self.label_encoder, open(save_dir / 'label_encoder.pkl', 'wb'))
        logger.info(f"Model saved to {save_dir}")
        
        return history
    
    def predict(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[str, float]:
        """
        Predict class for single sample.
        
        Args:
            amplitude: [temporal, antenna_pairs] tensor
            phase: [temporal, antenna_pairs] tensor
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        self.model.eval()
        
        # Add batch dimension
        amp_tensor = torch.FloatTensor(amplitude).unsqueeze(0).unsqueeze(0).to(self.device)
        phase_tensor = torch.FloatTensor(phase).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(amp_tensor, phase_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
        
        # Decode label
        predicted_class = self.label_encoder.inverse_transform([class_idx.item()])[0]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def save_model(self, save_path: str):
        """Save model checkpoint."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        logger.info(f"Model loaded from {load_path}")
