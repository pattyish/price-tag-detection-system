"""
Logger utility for structured logging throughout the system.
Supports console, file, JSON, and optional database logging.
"""

import logging
import logging.config
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


class PriceDetectionLogger:
    """Central logger configuration for Price Detection System."""
    
    _loggers = {}
    
    @classmethod
    def setup_logging(
        cls,
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_json: bool = True,
        enable_file: bool = True,
        enable_console: bool = True
    ) -> None:
        """
        Setup logging configuration for the entire system.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_json: Enable JSON log format
            enable_file: Enable file logging
            enable_console: Enable console logging
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{log_dir}/training").mkdir(parents=True, exist_ok=True)
        Path(f"{log_dir}/inference").mkdir(parents=True, exist_ok=True)
        Path(f"{log_dir}/errors").mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        simple_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        timestamp_format = '%Y-%m-%d %H:%M:%S'
        
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            console_formatter = ColoredFormatter(
                f'[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                datefmt=timestamp_format
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        if enable_file:
            # Standard log file
            log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt=timestamp_format
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        if enable_json:
            # JSON log file
            json_log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            json_handler.setLevel(getattr(logging, log_level))
            json_formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
            json_handler.setFormatter(json_formatter)
            root_logger.addHandler(json_handler)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Module name (typically __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]


# Convenience function for quick logger creation
def get_logger(name: str) -> logging.Logger:
    """Shorthand to get a logger instance."""
    return PriceDetectionLogger.get_logger(name)
