#!/usr/bin/env python
"""
Model Evaluation Module for Price Tag Detection
Handles model evaluation, metrics calculation, and performance analysis
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from ultralytics import YOLO
import numpy as np

from src.utils.logger import PriceDetectionLogger
from src.utils.config import PriceDetectionConfig

logger = PriceDetectionLogger().get_logger(__name__)


class ModelEvaluator:
    """Evaluate YOLO detection models and compute metrics"""
    
    def __init__(self, config: PriceDetectionConfig):
        """
        Initialize evaluator
        
        Args:
            config: PriceDetectionConfig object
        """
        self.config = config
        self.logger = logger
        self.results_dir = Path(config.training_config.get('results_dir', 'evaluation_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        img_size: int = 640,
        batch_size: int = 16,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset
        
        Args:
            model_path: Path to model weights
            dataset_path: Path to dataset directory
            img_size: Input image size
            batch_size: Batch size
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Evaluation results dictionary
        """
        try:
            self.logger.info(f"Loading model: {model_path}")
            model = YOLO(model_path)
            
            self.logger.info(f"Evaluating on dataset: {dataset_path}")
            
            # Prepare dataset YAML
            dataset_yaml = self._prepare_dataset_yaml(dataset_path)
            
            # Run validation
            results = model.val(
                data=dataset_yaml,
                imgsz=img_size,
                batch=batch_size,
                conf=confidence_threshold
            )
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            self.logger.info("Evaluation completed successfully")
            
            return {
                'success': True,
                'model': model_path,
                'dataset': dataset_path,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_models(
        self,
        model_paths: list,
        dataset_path: str
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset
        
        Args:
            model_paths: List of model weight paths
            dataset_path: Path to dataset
            
        Returns:
            Comparison results
        """
        try:
            self.logger.info(f"Comparing {len(model_paths)} models")
            
            comparison_results = []
            
            for model_path in model_paths:
                result = self.evaluate_model(
                    model_path=model_path,
                    dataset_path=dataset_path
                )
                if result['success']:
                    comparison_results.append({
                        'model': model_path,
                        'metrics': result['metrics']
                    })
            
            # Find best model
            best_model = max(
                comparison_results,
                key=lambda x: x['metrics'].get('mAP50', 0)
            )
            
            self.logger.info(f"Best model: {best_model['model']}")
            
            return {
                'success': True,
                'comparison': comparison_results,
                'best_model': best_model,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Model comparison failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def compute_per_class_metrics(
        self,
        model_path: str,
        dataset_path: str
    ) -> Dict[str, Any]:
        """
        Compute per-class performance metrics
        
        Args:
            model_path: Path to model weights
            dataset_path: Path to dataset
            
        Returns:
            Per-class metrics
        """
        try:
            self.logger.info("Computing per-class metrics")
            
            model = YOLO(model_path)
            dataset_yaml = self._prepare_dataset_yaml(dataset_path)
            
            results = model.val(data=dataset_yaml)
            
            # Extract per-class metrics
            per_class = {}
            if hasattr(results, 'results_dict'):
                for class_id, class_metrics in results.results_dict.items():
                    per_class[class_id] = {
                        'precision': class_metrics.get('precision'),
                        'recall': class_metrics.get('recall'),
                        'mAP50': class_metrics.get('mAP50')
                    }
            
            return {
                'success': True,
                'per_class_metrics': per_class
            }
        
        except Exception as e:
            self.logger.error(f"Per-class metrics computation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        report_name: str = 'evaluation_report.json'
    ) -> str:
        """
        Save evaluation results to JSON report
        
        Args:
            evaluation_results: Results dictionary
            report_name: Output report filename
            
        Returns:
            Path to saved report
        """
        try:
            report_path = self.results_dir / report_name
            
            with open(report_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            self.logger.info(f"Report saved: {report_path}")
            return str(report_path)
        
        except Exception as e:
            self.logger.error(f"Report saving failed: {str(e)}")
            return ""
    
    def _extract_metrics(self, results) -> Dict[str, Any]:
        """Extract key metrics from validation results"""
        metrics = {
            'mAP50': getattr(results.box, 'map50', None) if hasattr(results, 'box') else None,
            'mAP': getattr(results.box, 'map', None) if hasattr(results, 'box') else None,
            'precision': getattr(results.box, 'mp', None) if hasattr(results, 'box') else None,
            'recall': getattr(results.box, 'mr', None) if hasattr(results, 'box') else None
        }
        return metrics
    
    def _prepare_dataset_yaml(self, dataset_path: str) -> str:
        """Prepare dataset.yaml for evaluation"""
        dataset_path = Path(dataset_path)
        yaml_path = dataset_path / 'dataset.yaml'
        
        if not yaml_path.exists():
            import yaml
            yaml_content = {
                'path': str(dataset_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': 1,
                'names': ['price_tag']
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)
        
        return str(yaml_path)


def main():
    """Command-line interface for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO price detection models')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Load config
    config = PriceDetectionConfig.from_yaml(args.config)
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate or compare
    if args.compare:
        results = evaluator.compare_models(args.compare, args.dataset)
        print("\nModel Comparison Results:")
        print(json.dumps(results, indent=2, default=str))
    else:
        results = evaluator.evaluate_model(args.model, args.dataset)
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2, default=str))
    
    # Save report
    evaluator.save_evaluation_report(results)


if __name__ == '__main__':
    main()
