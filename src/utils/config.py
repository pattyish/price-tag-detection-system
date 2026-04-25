"""Configuration file for Price Detection System."""

import os
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """YOLO Model configuration."""
    model_name: str = "yolov8m"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    pretrained: bool = True
    device: str = "cuda"  # cuda or cpu
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    imgsz: int = 640
    batch_size: int = 16


@dataclass
class OCRConfig:
    """OCR Engine configuration."""
    engine: str = "easyocr"  # easyocr or tesseract
    languages: list = None
    gpu: bool = True
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data"
    train_dir: str = "retail_price_tag_data/train"
    val_dir: str = "retail_price_tag_data/val"
    test_dir: str = "retail_price_tag_data/test"
    raw_dir: str = "retail_price_tag_data/raw"
    processed_dir: str = "retail_price_tag_data/processed"
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    img_size: int = 640
    augment: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    patience: int = 20
    save_period: int = 10
    device: str = "cuda"
    workers: int = 4
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = "models/weights/best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_detections: int = 100
    imgsz: int = 640


@dataclass
class APIConfig:
    """API Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    reload: bool = True
    max_upload_size: int = 52428800  # 50MB


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = True
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class PriceDetectionConfig:
    """Main configuration class."""
    model: ModelConfig = None
    ocr: OCRConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    api: APIConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "PriceDetectionConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PriceDetectionConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'ocr' in config_dict:
            config.ocr = OCRConfig(**config_dict['ocr'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'inference' in config_dict:
            config.inference = InferenceConfig(**config_dict['inference'])
        if 'api' in config_dict:
            config.api = APIConfig(**config_dict['api'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
def get_default_config() -> PriceDetectionConfig:
    """Get default configuration."""
    return PriceDetectionConfig()
