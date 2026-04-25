"""
OCR Engine for Price Tag Text Extraction
Supports multiple OCR backends (EasyOCR, Tesseract)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import OCRConfig

logger = get_logger(__name__)


class OCREngine(Enum):
    """Supported OCR engines."""
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


@dataclass
class OCRResult:
    """Single OCR result."""
    text: str
    confidence: float
    bbox: Optional[List[List[float]]] = None  # Character-level bounding boxes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox
        }


class BaseOCREngine:
    """Base class for OCR engines."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from image."""
        raise NotImplementedError
    
    def extract_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Extract text from multiple images."""
        results = []
        for image in images:
            results.append(self.extract_text(image))
        return results


class EasyOCREngine(BaseOCREngine):
    """EasyOCR-based OCR engine."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.reader = None
        self._setup_reader()
        logger.info(f"EasyOCR initialized with languages: {config.languages}")
    
    def _setup_reader(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.gpu,
                verbose=False
            )
            logger.info("EasyOCR reader initialized successfully")
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            OCRResult object
        """
        try:
            # EasyOCR expects RGB, convert from BGR
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Extract text
            results = self.reader.readtext(image_rgb)
            
            if not results:
                logger.warning("No text detected in image")
                return OCRResult(text="", confidence=0.0, bbox=None)
            
            # Extract text and confidence
            extracted_text = " ".join([text[1] for text in results])
            avg_confidence = np.mean([text[2] for text in results])
            bboxes = [text[0] for text in results]
            
            logger.debug(f"Extracted text: '{extracted_text}' (confidence: {avg_confidence:.3f})")
            
            return OCRResult(
                text=extracted_text,
                confidence=avg_confidence,
                bbox=bboxes
            )
        except Exception as e:
            logger.error(f"Error during EasyOCR extraction: {e}")
            raise
    
    def extract_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Extract text from multiple images efficiently."""
        results = []
        logger.info(f"Processing {len(images)} images with EasyOCR")
        
        for idx, image in enumerate(images):
            try:
                result = self.extract_text(image)
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(images)} images")
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append(OCRResult(text="", confidence=0.0))
        
        return results


class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCR-based engine."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.pytesseract = None
        self._setup_tesseract()
        logger.info(f"Tesseract initialized")
    
    def _setup_tesseract(self) -> None:
        """Initialize Tesseract."""
        try:
            import pytesseract
            self.pytesseract = pytesseract
            logger.info("Pytesseract initialized successfully")
        except ImportError:
            logger.error("Pytesseract not installed. Install with: pip install pytesseract")
            logger.error("Also ensure Tesseract executable is installed on your system")
            raise
        except Exception as e:
            logger.error(f"Error initializing Tesseract: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text using Tesseract.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            OCRResult object
        """
        try:
            # Convert BGR to RGB for better OCR performance
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Extract text with config for numeric data
            config = '--oem 3 --psm 6'
            text = self.pytesseract.image_to_string(image_rgb, config=config)
            
            # Get detailed data
            data = self.pytesseract.image_to_data(image_rgb, output_type=self.pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['confidence'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            text = text.strip()
            logger.debug(f"Extracted text: '{text}' (confidence: {avg_confidence:.3f})")
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                bbox=None
            )
        except Exception as e:
            logger.error(f"Error during Tesseract extraction: {e}")
            raise
    
    def extract_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Extract text from multiple images."""
        results = []
        logger.info(f"Processing {len(images)} images with Tesseract")
        
        for idx, image in enumerate(images):
            try:
                result = self.extract_text(image)
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(images)} images")
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append(OCRResult(text="", confidence=0.0))
        
        return results


class OCRExtractor:
    """
    Main OCR extractor class.
    Provides unified interface for multiple OCR backends.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize OCR extractor.
        
        Args:
            config: OCRConfig instance
        """
        self.config = config
        self.engine = self._setup_engine()
        logger.info(f"OCRExtractor initialized with engine: {config.engine}")
    
    def _setup_engine(self) -> BaseOCREngine:
        """Setup the appropriate OCR engine."""
        try:
            if self.config.engine.lower() == "easyocr":
                return EasyOCREngine(self.config)
            elif self.config.engine.lower() == "tesseract":
                return TesseractOCREngine(self.config)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.config.engine}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine {self.config.engine}: {e}")
            raise
    
    def extract(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from image.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            OCRResult object
        """
        return self.engine.extract_text(image)
    
    def extract_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Extract text from multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of OCRResult objects
        """
        return self.engine.extract_batch(images)
    
    def switch_engine(self, engine_name: str) -> None:
        """
        Switch to a different OCR engine.
        
        Args:
            engine_name: Name of the engine (easyocr or tesseract)
        """
        try:
            self.config.engine = engine_name
            self.engine = self._setup_engine()
            logger.info(f"Switched to OCR engine: {engine_name}")
        except Exception as e:
            logger.error(f"Error switching OCR engine: {e}")
            raise
