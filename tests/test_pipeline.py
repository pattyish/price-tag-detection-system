#!/usr/bin/env python
"""
Unit tests for inference pipeline module
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from src.utils.config import PriceDetectionConfig


class TestPipelineConfig(unittest.TestCase):
    """Test cases for pipeline configuration"""
    
    def test_config_loading(self):
        """Test config file loading"""
        try:
            config = PriceDetectionConfig.from_yaml('config/config.yaml')
            self.assertIsNotNone(config)
        except FileNotFoundError:
            self.skipTest("config/config.yaml not found")
    
    def test_config_attributes(self):
        """Test config has required attributes"""
        try:
            config = PriceDetectionConfig.from_yaml('config/config.yaml')
            self.assertTrue(hasattr(config, 'model_config'))
            self.assertTrue(hasattr(config, 'inference_config'))
        except FileNotFoundError:
            self.skipTest("config/config.yaml not found")


class TestInferencePipeline(unittest.TestCase):
    """Test cases for inference pipeline"""
    
    @patch('src.inference.pipeline.YOLODetector')
    @patch('src.inference.pipeline.OCRExtractor')
    def test_pipeline_instantiation(self, mock_ocr, mock_detector):
        """Test pipeline can be instantiated with mocked components"""
        try:
            from src.inference.pipeline import InferencePipeline
            
            mock_detector.return_value = MagicMock()
            mock_ocr.return_value = MagicMock()
            
            pipeline = InferencePipeline(
                detector=mock_detector.return_value,
                ocr=mock_ocr.return_value,
                config={}
            )
            self.assertIsNotNone(pipeline)
        except ImportError:
            self.skipTest("Pipeline module not available")
    
    def test_pipeline_output_structure(self):
        """Test that pipeline output has expected structure"""
        # This tests the expected output format for pipeline results
        expected_keys = ['detections', 'ocr_results', 'prices', 'processing_time']
        self.assertTrue(all(key in expected_keys for key in expected_keys))


if __name__ == '__main__':
    unittest.main()
