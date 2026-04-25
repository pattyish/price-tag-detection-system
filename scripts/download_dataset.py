#!/usr/bin/env python
"""
Download and prepare datasets for training.
Usage: python scripts/download_dataset.py --dataset retail-ocr [--output data]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import PriceDetectionLogger
from src.utils.config import DataConfig
from src.data.dataset_loader import DatasetManager

logger = None


def main():
    global logger
    
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset name to download'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data',
        help='Output directory for dataset'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split dataset into train/val/test after download'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Training split ratio'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    PriceDetectionLogger.setup_logging(log_level=args.log_level)
    logger = PriceDetectionLogger.get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("PRICE DETECTION - DATASET DOWNLOADER")
    logger.info("=" * 80)
    
    try:
        # List available datasets
        if args.list_datasets:
            logger.info("Available datasets:")
            for name, info in DatasetManager.AVAILABLE_DATASETS.items():
                logger.info(f"  - {name}: {info['description']}")
                logger.info(f"    Source: {info['source']}")
            return
        
        # Setup data configuration
        config = DataConfig(data_dir=args.output)
        manager = DatasetManager(config)
        
        # Check if dataset already exists
        if manager.check_dataset(args.dataset):
            logger.warning(f"Dataset '{args.dataset}' already exists")
        else:
            logger.info(f"Downloading dataset: {args.dataset}")
            dataset_path = manager.download_dataset(args.dataset)
            logger.info(f"Dataset downloaded successfully to: {dataset_path}")
        
        # Split dataset if requested
        if args.split:
            logger.info("Organizing dataset into train/val/test splits...")
            dataset_path = Path(args.output) / 'raw' / args.dataset
            
            split_info = manager.organize_dataset(
                str(dataset_path),
                split_ratios=(args.train_split, args.val_split, 1.0 - args.train_split - args.val_split)
            )
            
            logger.info("Dataset split summary:")
            logger.info(f"  - Train: {split_info['train']} samples")
            logger.info(f"  - Val: {split_info['val']} samples")
            logger.info(f"  - Test: {split_info['test']} samples")
            logger.info(f"  - Total: {split_info['total']} samples")
        
        logger.info("=" * 80)
        logger.info("Dataset download complete!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
