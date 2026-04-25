#!/usr/bin/env python
"""
Unit tests for OCR engine module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.core.ocr_engine import OCRResult, BaseOCREngine


class TestOCRResult(unittest.TestCase):
    """Test cases for OCRResult"""
    
    def test_ocr_result_creation(self):
        """Test OCRResult dataclass creation"""
        result = OCRResult(
            text="$19.99",
            confidence=0.95,
            bbox=[50, 50, 150, 100]
        )
        self.assertEqual(result.text, "$19.99")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(len(result.bbox), 4)
    
    def test_ocr_result_empty_text(self):
        """Test OCR result with empty text"""
        result = OCRResult(
            text="",
            confidence=0.0,
            bbox=[0, 0, 0, 0]
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)
    
    def test_ocr_confidence_validation(self):
        """Test OCR confidence is between 0 and 1"""
        result = OCRResult(
            text="test",
            confidence=0.85,
            bbox=[0, 0, 100, 50]
        )
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestBaseOCREngine(unittest.TestCase):
    """Test cases for OCR engine interface"""
    
    def test_ocr_engine_instantiation(self):
        """Test that OCR engines can be instantiated"""
        try:
            # This will skip if OCR library not installed
            engine = BaseOCREngine()
            self.assertIsNotNone(engine)
        except ImportError:
            self.skipTest("OCR libraries not installed")


if __name__ == '__main__':
    unittest.main()
