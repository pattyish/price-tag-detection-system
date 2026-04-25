#!/usr/bin/env python
"""
YOLO Training Module for Price Tag Detection
Handles model training, validation, and checkpoint management
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from ultralytics import YOLO

from src.utils.logger import PriceDetectionLogger
from src.utils.config import PriceDetectionConfig

logger = PriceDetectionLogger().get_logger(__name__)


class YOLOTrainer:
    """Wrapper for YOLO training with custom callbacks and logging"""
    
    def __init__(self, config: PriceDetectionConfig):
        """
        Initialize trainer
        
        Args:
            config: PriceDetectionConfig object with training parameters
        """
        self.config = config
        self.logger = logger
        self.project_dir = Path(config.training_config.get('project_dir', 'runs/detect'))
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        model_name = config.model_config.get('name', 'yolov8n.pt')
        self.logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
    
    def prepare_dataset_yaml(self, dataset_path: str) -> str:
        """
        Prepare dataset YAML for training
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Path to generated dataset.yaml
        """
        dataset_path = Path(dataset_path)
        
        # Auto-detect train/val/test splits
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        test_dir = dataset_path / 'test'
        
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (1 for price tag)
            'names': ['price_tag']
        }
        
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.logger.info(f"Dataset YAML created: {yaml_path}")
        return str(yaml_path)
    
    def train(
        self,
        dataset_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        patience: int = 20,
        device: Optional[int] = 0,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Train YOLO model
        
        Args:
            dataset_path: Path to dataset directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            patience: Early stopping patience
            device: GPU device ID (None for CPU)
            resume: Resume from last checkpoint
            
        Returns:
            Training results dictionary
        """
        try:
            # Prepare dataset YAML
            dataset_yaml = self.prepare_dataset_yaml(dataset_path)
            
            self.logger.info(f"Starting training: {epochs} epochs, batch={batch_size}")
            
            # Run training
            results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                device=device,
                project=str(self.project_dir),
                name='price_detection',
                exist_ok=True,
                resume=resume,
                save=True,
                plots=True
            )
            
            self.logger.info("Training completed successfully")
            
            # Save best model
            best_model_path = self.project_dir / 'price_detection' / 'weights' / 'best.pt'
            if best_model_path.exists():
                self.logger.info(f"Best model saved: {best_model_path}")
            
            return {
                'success': True,
                'results': results,
                'model_path': str(best_model_path),
                'epochs': epochs,
                'batch_size': batch_size
            }
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate(
        self,
        dataset_path: str,
        model_path: Optional[str] = None,
        img_size: int = 640,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Validate trained model
        
        Args:
            dataset_path: Path to dataset directory
            model_path: Path to model weights (if None, uses current model)
            img_size: Input image size
            batch_size: Batch size for validation
            
        Returns:
            Validation results dictionary
        """
        try:
            if model_path:
                self.logger.info(f"Loading model for validation: {model_path}")
                model = YOLO(model_path)
            else:
                model = self.model
            
            # Prepare dataset YAML
            dataset_yaml = self.prepare_dataset_yaml(dataset_path)
            
            self.logger.info("Starting validation...")
            results = model.val(
                data=dataset_yaml,
                imgsz=img_size,
                batch=batch_size
            )
            
            self.logger.info("Validation completed successfully")
            
            return {
                'success': True,
                'results': results,
                'metrics': {
                    'map50': results.box.map50 if hasattr(results.box, 'map50') else None,
                    'map': results.box.map if hasattr(results.box, 'map') else None
                }
            }
        
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export(
        self,
        model_path: Optional[str] = None,
        format: str = 'onnx'
    ) -> Dict[str, Any]:
        """
        Export model to different formats
        
        Args:
            model_path: Path to model weights
            format: Export format ('onnx', 'torchscript', 'tflite', 'pb')
            
        Returns:
            Export result dictionary
        """
        try:
            if model_path:
                model = YOLO(model_path)
            else:
                model = self.model
            
            self.logger.info(f"Exporting model to {format} format...")
            export_path = model.export(format=format)
            
            self.logger.info(f"Model exported: {export_path}")
            
            return {
                'success': True,
                'export_path': str(export_path),
                'format': format
            }
        
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Command-line interface for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model for price detection')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = PriceDetectionConfig.from_yaml(args.config)
    
    # Create trainer
    trainer = YOLOTrainer(config)
    
    # Train
    results = trainer.train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        resume=args.resume
    )
    
    if results['success']:
        print(f"\n✓ Training completed successfully!")
        print(f"Model saved: {results['model_path']}")
    else:
        print(f"\n✗ Training failed: {results['error']}")


if __name__ == '__main__':
    main()
