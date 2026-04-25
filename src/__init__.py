"""Price Detection System - Main package."""

__version__ = "1.0.0"
__author__ = "Retail AI Team"

from src.utils.logger import PriceDetectionLogger, get_logger
from src.utils.config import PriceDetectionConfig
from src.core.yolo_detector import YOLODetector
from src.core.ocr_engine import OCRExtractor
from src.core.post_processor import PriceExtractor, PriceValidator
from src.inference.pipeline import InferencePipeline, PipelineBuilder

__all__ = [
    'PriceDetectionLogger',
    'get_logger',
    'PriceDetectionConfig',
    'YOLODetector',
    'OCRExtractor',
    'PriceExtractor',
    'PriceValidator',
    'InferencePipeline',
    'PipelineBuilder',
]
