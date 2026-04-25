#!/usr/bin/env python
"""
Training Script for YOLO Price Detection Model
Usage: python scripts/train.py --config config/config.yaml [--epochs 100] [--batch-size 16]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import PriceDetectionLogger
from src.utils.config import PriceDetectionConfig
from src.core.yolo_detector import YOLODetector

logger = None


def main():
    global logger
    
    parser = argparse.ArgumentParser(description='Train YOLO model for price detection')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to train on (cuda or cpu)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='YOLO model size (n, s, m, l, x)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='retail_price_tag_data/train',
        help='Path to training data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/weights',
        help='Output directory for trained weights'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    PriceDetectionLogger.setup_logging(log_level=args.log_level)
    logger = PriceDetectionLogger.get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("PRICE DETECTION - MODEL TRAINING")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        logger.info(f"Loading config from: {args.config}")
        config = PriceDetectionConfig.from_yaml(args.config)
        
        # Override with command line arguments
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.model:
            config.model.model_name = f"yolov8{args.model}"
        
        config.model.device = args.device
        config.training.device = args.device
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Model: {config.model.model_name}")
        logger.info(f"  - Epochs: {config.training.epochs}")
        logger.info(f"  - Batch size: {config.training.batch_size}")
        logger.info(f"  - Device: {config.model.device}")
        logger.info(f"  - Learning rate: {config.training.learning_rate}")
        
        # Initialize detector
        logger.info("Initializing YOLO detector...")
        detector = YOLODetector(config.model)
        
        # Check data directory
        data_path = Path(args.data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {args.data_dir}")
            logger.info("Please prepare your dataset first using: python scripts/download_dataset.py")
            sys.exit(1)
        
        logger.info(f"Training data directory: {args.data_dir}")
        
        # Train model
        logger.info("Starting model training...")
        logger.info("Note: Training requires YOLO data format (images + YOLO txt annotations)")
        
        # For actual training, you would use:
        # results = detector.model.train(
        #     data=str(data_path),
        #     epochs=config.training.epochs,
        #     batch=config.training.batch_size,
        #     device=config.model.device,
        #     ...
        # )
        
        logger.warning("Training requires a properly formatted dataset")
        logger.warning("Expected structure:")
        logger.warning("  retail_price_tag_data/")
        logger.warning("    images/")
        logger.warning("      *.jpg")
        logger.warning("    labels/")
        logger.warning("      *.txt  (YOLO format)")
        logger.warning("")
        logger.warning("YOLO format annotation (labels/*.txt):")
        logger.warning("  <class_id> <x_center> <y_center> <width> <height>")
        logger.warning("  (all coordinates normalized 0-1)")
        
        # Placeholder for actual training
        logger.info("")
        logger.info("To train the model with your dataset, ensure the data directory")
        logger.info("contains 'images/' and 'labels/' subdirectories with YOLO format annotations.")
        
        logger.info("=" * 80)
        logger.info("Training script ready!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
