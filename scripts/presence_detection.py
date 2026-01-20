"""
Presence Detection Pipeline Executable Script

Complete end-to-end pipeline for WiFi CSI-based presence/absence detection:
1. Load WiFi CSI data with RSSI, rate, noise, channel info
2. Expand features with statistical measures
3. Scale features using StandardScaler
4. Train Random Forest with GridSearchCV hyperparameter tuning
5. Evaluate on test set
6. Generate performance report

Usage:
    python presence_detection.py --data_file presence_data.csv --output_dir ./results --use_grid_search
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PresenceDetectionModel
from src.data_utils import DataValidator
from src.utils import setup_logging
from src.config import PRESENCE_MODEL_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WiFi CSI Presence Detection Pipeline'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to CSV file with WiFi CSI data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save models and results'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of data for testing'
    )
    
    parser.add_argument(
        '--use_grid_search',
        action='store_true',
        help='Use GridSearchCV for hyperparameter tuning'
    )
    
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
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
    """Run complete presence detection pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(
        log_file=str(output_dir / 'presence_detection.log'),
        level=args.log_level
    )
    
    logger.info("="*60)
    logger.info("WiFi CSI Presence Detection Pipeline")
    logger.info("="*60)
    
    try:
        # ===== STEP 1: Load Data =====
        logger.info("\n[STEP 1] Loading WiFi CSI data...")
        
        model = PresenceDetectionModel()
        X, y = model.load_data(args.data_file)
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Classes: {np.unique(y)}")
        
        # ===== STEP 2: Validate Data =====
        logger.info("\n[STEP 2] Validating data...")
        
        missing_check = DataValidator.check_missing_values(X)
        logger.info(f"Missing values check: {missing_check['valid']}")
        
        range_check = DataValidator.check_data_ranges(X)
        logger.info(f"Data range: [{range_check['min']:.2f}, {range_check['max']:.2f}]")
        
        balance_check = DataValidator.check_class_balance(y)
        logger.info(f"Class balance ratio: {balance_check['balance_ratio']:.2f}")
        logger.info(f"Class counts: {dict(zip(balance_check['classes'], balance_check['counts']))}")
        
        # ===== STEP 3: Train Model =====
        logger.info("\n[STEP 3] Training presence detection model...")
        
        if args.use_grid_search:
            logger.info("Using GridSearchCV for hyperparameter tuning...")
            metrics = model.train_with_tuning(
                X=X,
                y=y,
                test_size=args.test_size,
                cv_folds=args.cv_folds
            )
        else:
            logger.info("Training with default hyperparameters...")
            metrics = model.train(
                X=X,
                y=y,
                test_size=args.test_size,
                n_estimators=PRESENCE_MODEL_CONFIG['n_estimators']
            )
        
        # ===== STEP 4: Display Results =====
        logger.info("\n[STEP 4] Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        
        # ===== STEP 5: Save Model =====
        logger.info("\n[STEP 5] Saving model...")
        
        model.save_model(str(output_dir / 'models'))
        logger.info(f"Model saved to {output_dir / 'models'}")
        
        # ===== STEP 6: Generate Report =====
        logger.info("\n[STEP 6] Generating evaluation report...")
        
        report = model.generate_report(X, y)
        logger.info("\nClassification Report:")
        logger.info(report)
        
        # Save report
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"\nResults saved to {output_dir}")
        logger.info("="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
