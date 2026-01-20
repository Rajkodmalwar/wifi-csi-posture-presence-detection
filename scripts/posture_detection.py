"""
Posture Detection Pipeline Executable Script

Complete end-to-end pipeline for WiFi CSI-based posture detection:
1. Load raw CSI data from CSV
2. Preprocess: phase unwrap, normalization
3. Extract features: distance, AoA, 3D keypoints
4. Train CNN model with dual modality branches
5. Evaluate on test set
6. Visualize results

Usage:
    python posture_detection.py --data_file data.csv --output_dir ./results --epochs 100
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import CSIPreprocessor, FeatureExtractor
from src.model import CSIModelTrainer, CSIModelEvaluator
from src.data_utils import DataSplitter, DataValidator, DataStatistics
from src.utils import setup_logging, create_directories, plot_data_distribution
from src.config import CSI_CONFIG, POSTURE_MODEL_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WiFi CSI Posture Detection Pipeline'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to CSV file with raw CSI data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save models and results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for training (cpu or cuda)'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of data for testing'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Run complete posture detection pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(
        log_file=str(output_dir / 'posture_detection.log'),
        level=args.log_level
    )
    
    logger.info("="*60)
    logger.info("WiFi CSI Posture Detection Pipeline")
    logger.info("="*60)
    
    try:
        # ===== STEP 1: Load and Preprocess Data =====
        logger.info("\n[STEP 1] Loading and preprocessing CSI data...")
        
        preprocessor = CSIPreprocessor(
            subcarriers=CSI_CONFIG['num_subcarriers'],
            antenna_pairs=CSI_CONFIG['antenna_pairs'],
            temporal_samples=CSI_CONFIG['temporal_samples'],
            phase_filter_size=CSI_CONFIG['phase_filter_size']
        )
        
        amplitude_tensors, phase_tensors, labels = preprocessor.preprocess_csv(
            args.data_file
        )
        
        logger.info(f"Amplitude shape: {amplitude_tensors.shape}")
        logger.info(f"Phase shape: {phase_tensors.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        # Validate data
        amp_validation = DataValidator.check_missing_values(amplitude_tensors)
        logger.info(f"Amplitude validation: {amp_validation}")
        
        # ===== STEP 2: Data Statistics =====
        logger.info("\n[STEP 2] Computing data statistics...")
        
        amp_stats = DataStatistics.compute_tensor_statistics(amplitude_tensors)
        phase_stats = DataStatistics.compute_tensor_statistics(phase_tensors)
        
        logger.info(f"Amplitude stats: Mean={amp_stats['mean']:.4f}, Std={amp_stats['std']:.4f}")
        logger.info(f"Phase stats: Mean={phase_stats['mean']:.4f}, Std={phase_stats['std']:.4f}")
        
        # ===== STEP 3: Train Model =====
        logger.info("\n[STEP 3] Training posture detection model...")
        
        trainer = CSIModelTrainer(
            num_classes=POSTURE_MODEL_CONFIG['num_classes'],
            learning_rate=POSTURE_MODEL_CONFIG['learning_rate'],
            weight_decay=POSTURE_MODEL_CONFIG['weight_decay'],
            device=args.device
        )
        
        history = trainer.fit(
            amplitude_tensors=amplitude_tensors,
            phase_tensors=phase_tensors,
            labels=labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            save_dir=str(output_dir / 'checkpoints')
        )
        
        logger.info(f"Training complete. Final accuracy: {history['val_accuracy'][-1]:.4f}")
        
        # ===== STEP 4: Evaluate Model =====
        logger.info("\n[STEP 4] Evaluating model...")
        
        # Split data for evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.train_val_test_split(
            np.arange(len(amplitude_tensors)),
            labels,
            test_size=args.test_size
        )
        
        evaluator = CSIModelEvaluator(
            model=trainer.model,
            label_encoder=trainer.label_encoder,
            device=args.device
        )
        
        test_metrics = evaluator.evaluate(
            amplitude_tensors[X_test],
            phase_tensors[X_test],
            y_test
        )
        
        logger.info(f"Test Metrics:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        
        # Per-class metrics
        per_class_metrics = evaluator.compute_per_class_metrics(
            amplitude_tensors[X_test],
            phase_tensors[X_test],
            y_test
        )
        
        logger.info("\nPer-class metrics:")
        for cls, metrics in per_class_metrics.items():
            logger.info(f"  {cls}: Precision={metrics['precision']:.4f}, "
                       f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        # Save visualizations
        logger.info("\n[STEP 5] Saving visualizations...")
        
        evaluator.plot_confusion_matrix(
            amplitude_tensors[X_test],
            phase_tensors[X_test],
            y_test,
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        
        evaluator.plot_per_class_performance(
            per_class_metrics,
            save_path=str(output_dir / 'per_class_metrics.png')
        )
        
        logger.info(f"\nResults saved to {output_dir}")
        logger.info("="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
