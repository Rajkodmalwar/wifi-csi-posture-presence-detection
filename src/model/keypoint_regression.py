"""
3D Body Keypoint Regression Module

Predicts 3D Cartesian coordinates for body keypoints from CSI data.
Uses multi-layer neural network with MSE loss for coordinate regression.

Classes:
    KeypointNet: Neural network for keypoint regression
    KeypointTrainer: Training and evaluation manager
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
import pickle

logger = logging.getLogger(__name__)


class KeypointNet(nn.Module):
    """
    Neural network for 3D body keypoint regression.
    
    Architecture:
    - Input: [batch, features]
    - Dense layers: input → 256 → 128 → 64 → 3 (x, y, z coordinates)
    - Activation: ReLU + BatchNorm
    - Dropout for regularization
    
    Output: 3D Cartesian coordinates in meters
    """
    
    def __init__(
        self,
        input_size: int = 50,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3
    ):
        """
        Initialize keypoint network.
        
        Args:
            input_size: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(KeypointNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Build layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (3 coordinates)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, input_size] input tensor
            
        Returns:
            [batch, 3] output coordinates
        """
        return self.network(x)


class KeypointDataset(Dataset):
    """PyTorch Dataset for keypoint data."""
    
    def __init__(
        self,
        features: np.ndarray,
        keypoints: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            features: [num_samples, num_features] feature array
            keypoints: [num_samples, 3] keypoint coordinates
        """
        self.features = torch.FloatTensor(features)
        self.keypoints = torch.FloatTensor(keypoints)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.features[idx], self.keypoints[idx]


class KeypointTrainer:
    """
    Training manager for keypoint regression model.
    
    Handles training, validation, and prediction of 3D body keypoints.
    """
    
    def __init__(
        self,
        input_size: int = 50,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            input_size: Number of input features
            learning_rate: Initial learning rate
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model = KeypointNet(input_size=input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average MSE loss
        """
        self.model.train()
        total_loss = 0.0
        
        for features, keypoints in train_loader:
            features = features.to(self.device)
            keypoints = keypoints.to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, keypoints)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average MSE loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, keypoints in val_loader:
                features = features.to(self.device)
                keypoints = keypoints.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, keypoints)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(
        self,
        features: np.ndarray,
        keypoints: np.ndarray,
        epochs: int = 100,
        batch_size: int = 8,
        test_size: float = 0.2,
        save_dir: str = './checkpoints'
    ) -> dict:
        """
        Train model on full dataset.
        
        Args:
            features: Input features [num_samples, num_features]
            keypoints: Target keypoints [num_samples, 3]
            epochs: Number of epochs
            batch_size: Training batch size
            test_size: Fraction for test set
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        idx = np.arange(len(features))
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=42)
        
        # Create datasets and loaders
        train_dataset = KeypointDataset(features[train_idx], keypoints[train_idx])
        val_dataset = KeypointDataset(features[val_idx], keypoints[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_dir_path / 'keypoint_model.pth')
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info(f"Best model saved to {save_dir}")
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict keypoints for input features.
        
        Args:
            features: [num_samples, num_features] array
            
        Returns:
            [num_samples, 3] predicted coordinates
        """
        self.model.eval()
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        return predictions.cpu().numpy()
