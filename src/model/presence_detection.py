"""
WiFi CSI Presence Detection Module

Random Forest classifier for WiFi-based presence/absence detection.
Includes data processing, model training with hyperparameter tuning,
and evaluation metrics.

Classes:
    PresenceDetectionModel: Complete presence detection pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import pickle

logger = logging.getLogger(__name__)


class PresenceDetectionModel:
    """
    Complete WiFi CSI presence detection system using Random Forest.
    
    Pipeline:
    1. Load WiFi CSI data with RSSI, rate, noise, channel info
    2. Expand features with statistical measures
    3. Scale features using StandardScaler
    4. Train/tune Random Forest with GridSearchCV
    5. Evaluate on test set
    
    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        label_encoder: LabelEncoder for class labels
    """
    
    def __init__(self):
        """Initialize presence detection model."""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load WiFi CSI data from CSV.
        
        Expected columns: rssi, rate, noise_floor, channel, label
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extract features and labels
        feature_cols = [col for col in df.columns 
                       if col not in ['label', 'target', 'class']]
        
        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else df['target'].values
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        return X, y
    
    def expand_features(self, X: np.ndarray) -> np.ndarray:
        """
        Expand features with statistical measures.
        
        Computes: mean, std, min, max for each original feature
        
        Args:
            X: [num_samples, num_features] feature array
            
        Returns:
            [num_samples, expanded_features] expanded array
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        expanded = [X]
        
        # Add squared features
        expanded.append(X ** 2)
        
        # Add interaction terms
        if X.shape[1] > 1:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    expanded.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        # Concatenate all features
        X_expanded = np.hstack(expanded)
        
        logger.info(f"Expanded features from {X.shape[1]} to {X_expanded.shape[1]}")
        
        return X_expanded
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100
    ) -> Dict[str, float]:
        """
        Train Random Forest model.
        
        Args:
            X: Feature array
            y: Label array
            test_size: Fraction for test set
            random_state: Random seed
            n_estimators: Number of trees
            
        Returns:
            Dictionary with performance metrics
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Expand features
        X_expanded = self.expand_features(X)
        self.feature_names = [f'feat_{i}' for i in range(X_expanded.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_expanded, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_with_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train Random Forest with GridSearchCV hyperparameter tuning.
        
        Args:
            X: Feature array
            y: Label array
            test_size: Fraction for test set
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best metrics
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Expand features
        X_expanded = self.expand_features(X)
        self.feature_names = [f'feat_{i}' for i in range(X_expanded.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_expanded, y_encoded, test_size=test_size, random_state=42,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [3, 5, 7],
            'min_samples_leaf': [1, 2, 3]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=-1
        )
        
        logger.info("Running GridSearchCV...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Best model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict presence/absence for input samples.
        
        Args:
            X: [num_samples, num_features] feature array
            
        Returns:
            [num_samples] predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_expanded = self.expand_features(X)
        X_scaled = self.scaler.transform(X_expanded)
        
        predictions = self.model.predict(X_scaled)
        decoded = self.label_encoder.inverse_transform(predictions)
        
        return decoded
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: [num_samples, num_features] feature array
            
        Returns:
            [num_samples, num_classes] probability array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_expanded = self.expand_features(X)
        X_scaled = self.scaler.transform(X_expanded)
        
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, save_dir: str):
        """
        Save model, scaler, and encoder to files.
        
        Args:
            save_dir: Directory to save files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        pickle.dump(self.model, open(save_dir / 'presence_model.pkl', 'wb'))
        pickle.dump(self.scaler, open(save_dir / 'scaler.pkl', 'wb'))
        pickle.dump(self.label_encoder, open(save_dir / 'label_encoder.pkl', 'wb'))
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, save_dir: str):
        """
        Load model, scaler, and encoder from files.
        
        Args:
            save_dir: Directory with saved files
        """
        save_dir = Path(save_dir)
        
        self.model = pickle.load(open(save_dir / 'presence_model.pkl', 'rb'))
        self.scaler = pickle.load(open(save_dir / 'scaler.pkl', 'rb'))
        self.label_encoder = pickle.load(open(save_dir / 'label_encoder.pkl', 'rb'))
        
        logger.info(f"Model loaded from {save_dir}")
    
    def generate_report(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            X: Test feature array
            y: Test label array
            
        Returns:
            Formatted report string
        """
        y_pred = self.predict(X)
        
        report = classification_report(y, y_pred)
        
        return report
