"""
Posture Detection Examples

5 educational examples demonstrating the posture detection system:
1. Load and preprocess CSI data
2. Extract features from tensors
3. Train CNN model
4. Make predictions on new data
5. Visualize results
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import CSIPreprocessor, FeatureExtractor
from src.model import CSIModelTrainer, CSIModelEvaluator
from src.utils import setup_logging, plot_data_distribution
from src.config import CSI_CONFIG, POSTURE_MODEL_CONFIG

# Setup logging
logger = setup_logging(level='INFO')


def example_1_load_and_preprocess():
    """Example 1: Load and preprocess raw CSI data."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Load and Preprocess CSI Data")
    logger.info("="*60)
    
    # Initialize preprocessor with CSI configuration
    preprocessor = CSIPreprocessor(
        subcarriers=CSI_CONFIG['num_subcarriers'],
        antenna_pairs=CSI_CONFIG['antenna_pairs'],
        temporal_samples=CSI_CONFIG['temporal_samples']
    )
    
    logger.info("CSI Preprocessor Configuration:")
    logger.info(f"  Subcarriers: {CSI_CONFIG['num_subcarriers']}")
    logger.info(f"  Antenna Pairs: {CSI_CONFIG['antenna_pairs']}")
    logger.info(f"  Temporal Samples: {CSI_CONFIG['temporal_samples']}")
    
    # In a real scenario, load from CSV:
    # amplitude_tensors, phase_tensors, labels = preprocessor.preprocess_csv('data.csv')
    
    # For demo, create synthetic data
    amplitude_tensors = np.random.randn(100, 150, 4) + np.array([-40, -45, -42, -44])
    phase_tensors = np.random.randn(100, 150, 4) * 0.5
    labels = np.array(['standing'] * 50 + ['sitting'] * 50)
    
    logger.info(f"\nLoaded synthetic data:")
    logger.info(f"  Amplitude shape: {amplitude_tensors.shape}")
    logger.info(f"  Phase shape: {phase_tensors.shape}")
    logger.info(f"  Labels: {np.unique(labels)}")
    
    return amplitude_tensors, phase_tensors, labels


def example_2_extract_features():
    """Example 2: Extract physical features from CSI tensors."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Extract Features from Tensors")
    logger.info("="*60)
    
    # Load sample data from Example 1
    amplitude_tensors, phase_tensors, labels = example_1_load_and_preprocess()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        wavelength=CSI_CONFIG['wavelength'],
        antenna_spacing=CSI_CONFIG['antenna_spacing']
    )
    
    logger.info("Extracting features...")
    
    # Extract features from first sample
    features = extractor.extract_all_features(
        amplitude_tensors[0],
        phase_tensors[0]
    )
    
    logger.info(f"\nExtracted features for sample 0:")
    for key, value in features.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Extract batch features
    features_df = extractor.extract_batch_features(
        amplitude_tensors[:10],
        phase_tensors[:10],
        labels[:10]
    )
    
    logger.info(f"\nBatch features shape: {features_df.shape}")
    logger.info(f"Feature columns: {list(features_df.columns)[:5]}...")  # Show first 5
    
    return features_df


def example_3_train_model():
    """Example 3: Train posture detection CNN model."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Train Posture Detection Model")
    logger.info("="*60)
    
    # Load data from Example 1
    amplitude_tensors, phase_tensors, labels = example_1_load_and_preprocess()
    
    # Initialize trainer
    trainer = CSIModelTrainer(
        num_classes=POSTURE_MODEL_CONFIG['num_classes'],
        learning_rate=POSTURE_MODEL_CONFIG['learning_rate'],
        device='cpu'
    )
    
    logger.info("Training model with configuration:")
    logger.info(f"  Classes: {POSTURE_MODEL_CONFIG['num_classes']}")
    logger.info(f"  Learning Rate: {POSTURE_MODEL_CONFIG['learning_rate']}")
    logger.info(f"  Batch Size: {POSTURE_MODEL_CONFIG['batch_size']}")
    
    # Encode labels (use 7 default classes)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform([labels[i] if isinstance(labels[i], str) else f'class_{labels[i] % 7}' 
                                       for i in range(len(labels))])
    
    # Train model
    history = trainer.fit(
        amplitude_tensors=amplitude_tensors,
        phase_tensors=phase_tensors,
        labels=labels_encoded,
        epochs=10,  # Reduced for demo
        batch_size=8,
        test_size=0.2,
        save_dir='./demo_checkpoints'
    )
    
    logger.info(f"\nTraining completed:")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")


def example_4_predict():
    """Example 4: Make predictions on new data."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Make Predictions")
    logger.info("="*60)
    
    # Load data and train model
    amplitude_tensors, phase_tensors, labels = example_1_load_and_preprocess()
    
    trainer = CSIModelTrainer(
        num_classes=POSTURE_MODEL_CONFIG['num_classes'],
        device='cpu'
    )
    
    # Quick training
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform([f'class_{i % 7}' for i in range(len(labels))])
    
    trainer.fit(
        amplitude_tensors=amplitude_tensors,
        phase_tensors=phase_tensors,
        labels=labels_encoded,
        epochs=5,
        save_dir='./demo_checkpoints'
    )
    
    # Make prediction on new sample
    logger.info("\nPredicting on new sample...")
    
    predicted_class, confidence = trainer.predict(
        amplitude_tensors[0],
        phase_tensors[0]
    )
    
    logger.info(f"  Predicted class: {predicted_class}")
    logger.info(f"  Confidence: {confidence:.4f}")
    
    # Batch prediction
    predictions, probabilities = trainer.model(
        __import__('torch').FloatTensor(amplitude_tensors[:5]).unsqueeze(1),
        __import__('torch').FloatTensor(phase_tensors[:5]).unsqueeze(1)
    )
    
    logger.info(f"\nBatch predictions shape: {predictions.shape}")
    logger.info(f"Sample probabilities shape: {probabilities.shape if hasattr(probabilities, 'shape') else 'N/A'}")


def example_5_visualize():
    """Example 5: Visualize data and results."""
    
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Visualize Data and Results")
    logger.info("="*60)
    
    # Load data
    amplitude_tensors, phase_tensors, labels = example_1_load_and_preprocess()
    
    logger.info("Visualization capabilities:")
    logger.info("  - plot_tensor_sample: Visualize amplitude and phase heatmaps")
    logger.info("  - plot_data_distribution: Show class distribution")
    logger.info("  - plot_confusion_matrix: Evaluate model performance")
    logger.info("  - plot_per_class_metrics: Show per-class metrics")
    
    # Example: Analyze data distribution
    logger.info("\nClass distribution:")
    unique_labels = np.unique([f'class_{i % 7}' for i in range(len(labels))])
    label_encoded = [i % 7 for i in range(len(labels))]
    
    for cls_id, cls_name in enumerate(unique_labels):
        count = sum(1 for x in label_encoded if x == cls_id)
        logger.info(f"  {cls_name}: {count} samples")
    
    logger.info("\nVisualization example calls:")
    logger.info("  from src.utils import plot_tensor_sample")
    logger.info("  plot_tensor_sample(amplitude_tensors[0], phase_tensors[0], save_path='sample.png')")


if __name__ == '__main__':
    """Run all examples."""
    
    logger.info("\n\n")
    logger.info("#"*60)
    logger.info("# WiFi CSI Posture Detection - Educational Examples")
    logger.info("#"*60)
    
    try:
        example_1_load_and_preprocess()
        example_2_extract_features()
        example_3_train_model()
        example_4_predict()
        example_5_visualize()
        
        logger.info("\n\n")
        logger.info("#"*60)
        logger.info("# All Examples Completed Successfully!")
        logger.info("#"*60)
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}", exc_info=True)
