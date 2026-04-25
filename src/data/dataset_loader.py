"""
Dataset Management and Preprocessing
Handles download, loading, and preprocessing of price detection datasets.
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import DataConfig

logger = get_logger(__name__)


class DatasetManager:
    """Manage dataset download and organization."""
    
    # Available datasets
    AVAILABLE_DATASETS = {
        'retail-ocr': {
            'name': 'Retail OCR Dataset',
            'source': 'huggingface',
            'url': 'https://huggingface.co/datasets/CCI-Digital/retail-price-tags',
            'description': 'Price tags from retail stores'
        },
        'synthetic-prices': {
            'name': 'Synthetic Price Tags',
            'source': 'local',
            'description': 'Synthetically generated price tag images'
        }
    }
    
    def __init__(self, config: DataConfig):
        """
        Initialize dataset manager.
        
        Args:
            config: DataConfig instance
        """
        self.config = config
        self._ensure_directories()
        logger.info("DatasetManager initialized")
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.config.data_dir,
            self.config.raw_dir,
            self.config.processed_dir,
            self.config.train_dir,
            self.config.val_dir,
            self.config.test_dir,
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def check_dataset(self, dataset_name: str) -> bool:
        """
        Check if dataset exists locally.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            True if dataset exists
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_path = Path(self.config.raw_dir) / dataset_name
        exists = dataset_path.exists()
        
        if exists:
            logger.info(f"Dataset '{dataset_name}' found at {dataset_path}")
        else:
            logger.info(f"Dataset '{dataset_name}' not found")
        
        return exists
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download dataset from source.
        
        Args:
            dataset_name: Name of dataset to download
            force: Force re-download even if exists
            
        Returns:
            Path to downloaded dataset
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.AVAILABLE_DATASETS[dataset_name]
        dataset_path = Path(self.config.raw_dir) / dataset_name
        
        if dataset_path.exists() and not force:
            logger.info(f"Dataset already exists at {dataset_path}")
            return dataset_path
        
        logger.info(f"Starting download of '{dataset_name}'...")
        logger.info(f"Dataset source: {dataset_info['source']}")
        
        if dataset_info['source'] == 'huggingface':
            return self._download_from_huggingface(dataset_name, dataset_path)
        elif dataset_info['source'] == 'local':
            logger.warning("This is a local dataset. Please add images manually.")
            dataset_path.mkdir(parents=True, exist_ok=True)
            return dataset_path
        else:
            raise ValueError(f"Unknown source: {dataset_info['source']}")
    
    def _download_from_huggingface(self, dataset_name: str, target_path: Path) -> Path:
        """
        Download dataset from Hugging Face.
        
        Args:
            dataset_name: Dataset identifier
            target_path: Target download path
            
        Returns:
            Path to downloaded dataset
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Downloading from Hugging Face: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Save dataset
            dataset.save_to_disk(str(target_path))
            logger.info(f"Dataset saved to {target_path}")
            
            return target_path
        
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error downloading from Hugging Face: {e}")
            raise
    
    def organize_dataset(self, dataset_path: str, split_ratios: Optional[Tuple] = None) -> Dict:
        """
        Organize dataset into train/val/test splits.
        
        Args:
            dataset_path: Path to dataset directory
            split_ratios: Tuple of (train, val, test) ratios
            
        Returns:
            Dictionary with split information
        """
        if split_ratios is None:
            split_ratios = (
                self.config.train_split,
                self.config.val_split,
                self.config.test_split
            )
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = sorted([
            f for f in dataset_path.rglob('*')
            if f.suffix.lower() in image_extensions
        ])
        
        logger.info(f"Found {len(images)} images in dataset")
        
        # Handle annotation files
        annotations = {}
        annotation_extensions = {'.txt', '.json', '.xml'}
        for image_file in images:
            annotation_candidates = [
                image_file.with_suffix(ext) for ext in annotation_extensions
            ]
            for ann_file in annotation_candidates:
                if ann_file.exists():
                    annotations[image_file.name] = ann_file
                    break
        
        logger.info(f"Found {len(annotations)} annotation files")
        
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # First split: train vs (val + test)
        train_images, temp_images = train_test_split(
            images,
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Second split: val vs test
        val_images, test_images = train_test_split(
            temp_images,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42
        )
        
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        # Copy files to splits
        for split_name, split_images in splits.items():
            split_dir = Path(getattr(self.config, f'{split_name}_dir'))
            split_dir.mkdir(parents=True, exist_ok=True)
            
            images_dir = split_dir / 'images'
            annotations_dir = split_dir / 'labels'
            images_dir.mkdir(exist_ok=True)
            annotations_dir.mkdir(exist_ok=True)
            
            for image_file in split_images:
                # Copy image
                import shutil
                shutil.copy(
                    image_file,
                    images_dir / image_file.name
                )
                
                # Copy annotation if exists
                if image_file.name in annotations:
                    shutil.copy(
                        annotations[image_file.name],
                        annotations_dir / annotations[image_file.name].name
                    )
            
            logger.info(f"Copied {len(split_images)} images to {split_name} split")
        
        return {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
            'total': len(images)
        }


class DataPreprocessor:
    """Preprocess images for model training."""
    
    def __init__(self, target_size: int = 640, augment: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size
            augment: Enable augmentation
        """
        self.target_size = target_size
        self.augment = augment
        logger.info(f"DataPreprocessor initialized: size={target_size}, augment={augment}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image array
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Resize
            image = cv2.resize(image, (self.target_size, self.target_size))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Augmentation
            if self.augment:
                image = self._augment_image(image)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        try:
            import albumentations as A
            
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=10, p=0.5),
                A.GaussNoise(p=0.2),
            ])
            
            # Convert normalized image back to 0-255 for augmentation
            image_uint8 = (image * 255).astype(np.uint8)
            augmented = transform(image=image_uint8)
            
            # Normalize back
            return augmented['image'].astype(np.float32) / 255.0
        
        except ImportError:
            logger.warning("albumentations not installed, skipping augmentation")
            return image
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Array of preprocessed images (N, H, W, C)
        """
        images = []
        for idx, image_path in enumerate(image_paths):
            image = self.preprocess_image(image_path)
            images.append(image)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Preprocessed {idx + 1}/{len(image_paths)} images")
        
        return np.array(images)
