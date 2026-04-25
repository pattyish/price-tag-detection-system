"""
YOLO-based Price Tag Detection Module
Detects price tag bounding boxes in retail shelf images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from ultralytics import YOLO
from src.utils.logger import get_logger
from src.utils.config import ModelConfig


logger = get_logger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int = 0
    class_name: str = "price_tag"
    
    def get_crop_coords(self) -> Tuple[int, int, int, int]:
        """Get crop coordinates for PIL/OpenCV."""
        return (int(self.x_min), int(self.y_min), int(self.x_max), int(self.y_max))
    
    def get_area(self) -> float:
        """Calculate bounding box area."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'width': self.x_max - self.x_min,
            'height': self.y_max - self.y_min,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
        }


class YOLODetector:
    """
    YOLO-based detector for price tags in retail images.
    Uses YOLOv8 for fast and accurate object detection.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize YOLO detector.
        
        Args:
            config: ModelConfig instance with detection parameters
        """
        self.config = config
        self.model = None
        self.device = config.device
        self._setup_model()
        logger.info(f"YOLODetector initialized with model: {config.model_name} on device: {self.device}")
    
    def _setup_model(self) -> None:
        """Load or initialize the YOLO model."""
        try:
            logger.info(f"Loading YOLO model: {self.config.model_name}")
            self.model = YOLO(f"{self.config.model_name}.pt")
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect price tags in image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of Detection objects
        """
        try:
            # Run inference
            start_time = time.time()
            results = self.model(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for idx in range(len(boxes)):
                        # Extract bounding box coordinates
                        xyxy = boxes.xyxy[idx].cpu().numpy()
                        confidence = boxes.conf[idx].cpu().item()
                        class_id = int(boxes.cls[idx].cpu().item())
                        
                        detection = Detection(
                            x_min=float(xyxy[0]),
                            y_min=float(xyxy[1]),
                            x_max=float(xyxy[2]),
                            y_max=float(xyxy[3]),
                            confidence=float(confidence),
                            class_id=class_id,
                            class_name=result.names.get(class_id, "unknown")
                        )
                        detections.append(detection)
            
            logger.debug(f"Detection complete: {len(detections)} tags detected in {inference_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def detect_from_path(self, image_path: str) -> List[Detection]:
        """
        Detect price tags from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of Detection objects
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            return self.detect(image)
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            raise
    
    def detect_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[List[Detection]]:
        """
        Batch process multiple images.
        
        Args:
            images: List of image arrays
            batch_size: Batch size for processing
            
        Returns:
            List of detection lists (one per image)
        """
        results = []
        total_images = len(images)
        
        for i in range(0, total_images, batch_size):
            batch = images[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total_images)}/{total_images})")
            
            try:
                # Run batch inference
                batch_results = self.model(
                    batch,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    verbose=False
                )
                
                # Process each result
                for result in batch_results:
                    detections = []
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        
                        for idx in range(len(boxes)):
                            xyxy = boxes.xyxy[idx].cpu().numpy()
                            confidence = boxes.conf[idx].cpu().item()
                            class_id = int(boxes.cls[idx].cpu().item())
                            
                            detection = Detection(
                                x_min=float(xyxy[0]),
                                y_min=float(xyxy[1]),
                                x_max=float(xyxy[2]),
                                y_max=float(xyxy[3]),
                                confidence=float(confidence),
                                class_id=class_id,
                                class_name=result.names.get(class_id, "unknown")
                            )
                            detections.append(detection)
                    
                    results.append(detections)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                raise
        
        return results
    
    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: float = 0.5,
        min_area: float = 100,
        max_area: Optional[float] = None
    ) -> List[Detection]:
        """
        Filter detections based on criteria.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            # Confidence check
            if det.confidence < min_confidence:
                continue
            
            # Area check
            area = det.get_area()
            if area < min_area:
                continue
            
            if max_area and area > max_area:
                continue
            
            filtered.append(det)
        
        logger.debug(f"Filtered detections: {len(detections)} -> {len(filtered)}")
        return filtered
    
    def nms(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.4
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of detections after NMS
        """
        if not detections:
            return []
        
        # Convert to format for NMS
        boxes = np.array([det.get_crop_coords() for det in detections])
        confidences = np.array([det.confidence for det in detections])
        
        # Sort by confidence
        sorted_indices = np.argsort(-confidences)
        
        keep = []
        while len(sorted_indices) > 0:
            keep.append(sorted_indices[0])
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[sorted_indices[0]]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_ious(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]
        
        result = [detections[i] for i in keep]
        logger.debug(f"NMS: {len(detections)} -> {len(result)} detections")
        return result
    
    @staticmethod
    def _calculate_ious(box: Tuple, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between one box and multiple boxes.
        
        Args:
            box: Single box (x1, y1, x2, y2)
            boxes: Multiple boxes array (N, 4)
            
        Returns:
            IoU array
        """
        x1, y1, x2, y2 = box
        
        xs1 = np.maximum(x1, boxes[:, 0])
        ys1 = np.maximum(y1, boxes[:, 1])
        xs2 = np.minimum(x2, boxes[:, 2])
        ys2 = np.minimum(y2, boxes[:, 3])
        
        inter_w = np.maximum(xs2 - xs1, 0)
        inter_h = np.maximum(ys2 - ys1, 0)
        inter = inter_w * inter_h
        
        union = (x2 - x1) * (y2 - y1) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
        ious = inter / (union + 1e-6)
        
        return ious
    
    def save_model(self, save_path: str) -> None:
        """Save model weights."""
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """Load model weights."""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
