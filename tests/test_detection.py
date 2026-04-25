#!/usr/bin/env python
"""
Unit tests for YOLO detection module
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.core.yolo_detector import YOLODetector, Detection


class TestYOLODetector(unittest.TestCase):
    """Test cases for YOLODetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'name': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.45,
            'device': 'cpu'
        }
    
    @patch('src.core.yolo_detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization"""
        detector = YOLODetector(self.config)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.confidence_threshold, 0.5)
    
    @patch('src.core.yolo_detector.YOLO')
    def test_detection_creation(self, mock_yolo):
        """Test Detection dataclass creation"""
        detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            class_id=0,
            class_name='price_tag'
        )
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(len(detection.bbox), 4)
    
    def test_detection_area_calculation(self):
        """Test bounding box area calculation"""
        detection = Detection(
            bbox=[0, 0, 100, 100],
            confidence=0.9,
            class_id=0
        )
        # Area should be (100-0) * (100-0) = 10000
        area = (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1])
        self.assertEqual(area, 10000)


if __name__ == '__main__':
    unittest.main()
