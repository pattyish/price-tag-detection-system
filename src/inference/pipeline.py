"""
End-to-End Inference Pipeline
Combines YOLO detection, OCR extraction, and post-processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import json
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.config import (
    PriceDetectionConfig, 
    ModelConfig, 
    OCRConfig,
    InferenceConfig
)
from src.core.yolo_detector import YOLODetector, Detection
from src.core.ocr_engine import OCRExtractor
from src.core.post_processor import PriceExtractor, PriceValidator, PriceData

logger = get_logger(__name__)


class InferencePipeline:
    """
    Complete inference pipeline for price tag detection and extraction.
    Combines YOLO detection, OCR, and price extraction.
    """
    
    def __init__(self, config: PriceDetectionConfig):
        """
        Initialize inference pipeline.
        
        Args:
            config: PriceDetectionConfig instance
        """
        self.config = config
        self.detector = None
        self.ocr = None
        self.extractor = None
        self.validator = None
        
        self._setup_components()
        logger.info("InferencePipeline initialized")
    
    def _setup_components(self) -> None:
        """Setup detection, OCR, and extraction components."""
        try:
            logger.info("Setting up detection component...")
            self.detector = YOLODetector(self.config.model)
            
            logger.info("Setting up OCR component...")
            self.ocr = OCRExtractor(self.config.ocr)
            
            logger.info("Setting up extraction component...")
            self.extractor = PriceExtractor(
                min_price=0.0,
                max_price=10000.0
            )
            
            logger.info("Setting up validator...")
            self.validator = PriceValidator()
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up components: {e}")
            raise
    
    def process_image(self, image_path: str, expected_price: Optional[float] = None) -> Dict:
        """
        Process single image through complete pipeline.
        
        Args:
            image_path: Path to image
            expected_price: Optional expected price for validation
            
        Returns:
            Dictionary with complete results
        """
        try:
            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            results = {
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'image_size': image.shape,
                'detections': [],
                'summary': {}
            }
            
            # Step 1: Detection
            logger.info("Step 1: Detecting price tags...")
            detections = self.detector.detect(image)
            logger.info(f"Detected {len(detections)} price tags")
            
            if len(detections) == 0:
                results['summary']['total_detections'] = 0
                results['summary']['valid_extractions'] = 0
                logger.warning("No price tags detected in image")
                return results
            
            # Step 2: OCR and extraction for each detection
            logger.info("Step 2: Extracting text from detections...")
            for idx, detection in enumerate(detections):
                detection_result = self._process_detection(
                    image, detection, idx, expected_price
                )
                results['detections'].append(detection_result)
            
            # Summary
            valid_count = sum(1 for d in results['detections'] if d['price']['is_valid'])
            results['summary'] = {
                'total_detections': len(detections),
                'valid_extractions': valid_count,
                'extraction_rate': valid_count / len(detections) if len(detections) > 0 else 0
            }
            
            logger.info(f"Pipeline complete: {valid_count}/{len(detections)} valid prices extracted")
            return results
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _process_detection(self, 
                          image: np.ndarray, 
                          detection: Detection, 
                          idx: int,
                          expected_price: Optional[float] = None) -> Dict:
        """
        Process a single detection.
        
        Args:
            image: Full image
            detection: Detection object
            idx: Detection index
            expected_price: Optional expected price
            
        Returns:
            Detection result dictionary
        """
        try:
            # Crop region around detection
            x1, y1, x2, y2 = detection.get_crop_coords()
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                logger.warning(f"Detection {idx}: Empty crop region")
                return {
                    'detection_idx': idx,
                    'detection': asdict(detection),
                    'ocr_result': None,
                    'price': None,
                    'validation': None,
                    'error': "Empty crop region"
                }
            
            # OCR extraction
            ocr_result = self.ocr.extract(cropped)
            
            # Price extraction
            price_data = self.extractor.extract_price(
                ocr_result.text,
                ocr_result.confidence
            )
            
            # Validation
            if expected_price:
                self.validator.expected_price = expected_price
            validation_result = self.validator.validate(price_data)
            
            return {
                'detection_idx': idx,
                'detection': asdict(detection),
                'ocr_result': ocr_result.to_dict(),
                'price': price_data.to_dict(),
                'validation': validation_result,
                'error': None
            }
        
        except Exception as e:
            logger.error(f"Error processing detection {idx}: {e}")
            return {
                'detection_idx': idx,
                'error': str(e)
            }
    
    def process_batch(self, 
                     image_paths: List[str],
                     expected_prices: Optional[List[float]] = None) -> List[Dict]:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image paths
            expected_prices: Optional list of expected prices
            
        Returns:
            List of result dictionaries
        """
        results = []
        expected_prices = expected_prices or [None] * len(image_paths)
        
        logger.info(f"Processing batch of {len(image_paths)} images...")
        
        for idx, (image_path, expected_price) in enumerate(zip(image_paths, expected_prices)):
            logger.info(f"Processing {idx + 1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_image(image_path, expected_price)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        logger.info(f"Batch processing complete: {len(results)} images processed")
        return results
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save pipeline results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON file
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def load_results(self, results_path: str) -> Dict:
        """
        Load pipeline results from JSON file.
        
        Args:
            results_path: Path to results file
            
        Returns:
            Results dictionary
        """
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded from {results_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise


class PipelineBuilder:
    """Builder for creating pipeline instances with custom configurations."""
    
    @staticmethod
    def create_default() -> InferencePipeline:
        """Create pipeline with default configuration."""
        config = PriceDetectionConfig()
        return InferencePipeline(config)
    
    @staticmethod
    def create_from_config_file(config_path: str) -> InferencePipeline:
        """Create pipeline from YAML config file."""
        config = PriceDetectionConfig.from_yaml(config_path)
        return InferencePipeline(config)
    
    @staticmethod
    def create_custom(
        model_name: str = "yolov8m",
        ocr_engine: str = "easyocr",
        device: str = "cuda",
        confidence_threshold: float = 0.5
    ) -> InferencePipeline:
        """
        Create pipeline with custom configuration.
        
        Args:
            model_name: YOLO model name
            ocr_engine: OCR engine to use
            device: Compute device (cuda or cpu)
            confidence_threshold: Detection confidence threshold
            
        Returns:
            InferencePipeline instance
        """
        config = PriceDetectionConfig()
        config.model.model_name = model_name
        config.model.device = device
        config.model.confidence_threshold = confidence_threshold
        config.ocr.engine = ocr_engine
        
        return InferencePipeline(config)
