"""
Post-Processing for Price Extraction and Validation
Parses OCR text to extract price values and validate data.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Currency(Enum):
    """Supported currencies."""
    USD = "$"
    EUR = "€"
    GBP = "£"
    JPY = "¥"
    GENERIC = ""


@dataclass
class PriceData:
    """Extracted price information."""
    price_value: Optional[float] = None
    currency: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0
    is_valid: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'price_value': self.price_value,
            'currency': self.currency,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'formatted_price': self.formatted_price()
        }
    
    def formatted_price(self) -> str:
        """Return formatted price string."""
        if self.price_value is None:
            return "Invalid"
        currency_str = self.currency if self.currency else ""
        return f"{currency_str}{self.price_value:.2f}"


class PriceExtractor:
    """Extract and validate price information from OCR text."""
    
    # Common currency symbols
    CURRENCY_PATTERNS = {
        Currency.USD: r'\$',
        Currency.EUR: r'€',
        Currency.GBP: r'£',
        Currency.JPY: r'¥|¥',
    }
    
    # Price patterns
    PRICE_PATTERNS = [
        # Format: $12.99, €15.50, etc.
        r'[$€£¥]?\s*(\d+(?:[.,]\d{2})?)',
        # Format: 12.99USD, 15.50GBP, etc.
        r'(\d+(?:[.,]\d{2})?)\s*(?:USD|EUR|GBP|JPY)?',
        # Format: price: 19.99
        r'(?:price|cost|amount)[:=\s]+(\d+(?:[.,]\d{2})?)',
    ]
    
    def __init__(self, 
                 min_price: float = 0.0,
                 max_price: float = 10000.0,
                 allow_currencies: List[str] = None):
        """
        Initialize price extractor.
        
        Args:
            min_price: Minimum allowed price value
            max_price: Maximum allowed price value
            allow_currencies: List of allowed currencies (None = all)
        """
        self.min_price = min_price
        self.max_price = max_price
        self.allow_currencies = allow_currencies or ["$", "€", "£", "¥"]
        logger.info(f"PriceExtractor initialized: min={min_price}, max={max_price}")
    
    def extract_price(self, text: str, confidence: float = 1.0) -> PriceData:
        """
        Extract price from OCR text.
        
        Args:
            text: Raw OCR text
            confidence: OCR confidence score
            
        Returns:
            PriceData object
        """
        logger.debug(f"Extracting price from text: '{text}' (conf: {confidence:.3f})")
        
        result = PriceData(raw_text=text, confidence=confidence)
        
        if not text or not text.strip():
            result.error_message = "Empty text"
            logger.warning("Empty text provided for price extraction")
            return result
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Try to extract currency
        currency = self._extract_currency(cleaned_text)
        result.currency = currency
        
        # Try to extract price value
        price_value = self._extract_price_value(cleaned_text)
        
        if price_value is None:
            result.error_message = "Could not extract price value"
            logger.warning(f"Failed to extract price from: '{text}'")
            return result
        
        # Validate price
        if not self._validate_price(price_value):
            result.error_message = f"Price {price_value} outside range [{self.min_price}, {self.max_price}]"
            logger.warning(f"Price {price_value} out of valid range")
            return result
        
        result.price_value = price_value
        result.is_valid = True
        logger.info(f"Successfully extracted price: {result.formatted_price()} (confidence: {confidence:.3f})")
        
        return result
    
    def extract_batch(self, texts: List[str], confidences: List[float] = None) -> List[PriceData]:
        """
        Extract prices from multiple texts.
        
        Args:
            texts: List of OCR texts
            confidences: List of confidence scores (optional)
            
        Returns:
            List of PriceData objects
        """
        if confidences is None:
            confidences = [1.0] * len(texts)
        
        results = []
        for text, conf in zip(texts, confidences):
            results.append(self.extract_price(text, conf))
        
        return results
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize OCR text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Replace common OCR mistakes
        replacements = {
            'O': '0',  # O -> 0
            'l': '1',  # lowercase L -> 1
            '|': '1',  # pipe -> 1
            'S': '5',  # S -> 5
            'Z': '2',  # Z -> 2
            'B': '8',  # B -> 8
        }
        
        for char, replacement in replacements.items():
            # Only replace in numeric contexts
            text = re.sub(f'({char})(?=[0-9])', replacement, text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def _extract_currency(text: str) -> Optional[str]:
        """
        Extract currency symbol from text.
        
        Args:
            text: Input text
            
        Returns:
            Currency symbol or None
        """
        # Check for currency symbols
        currency_pattern = r'[$€£¥]'
        match = re.search(currency_pattern, text)
        if match:
            return match.group()
        
        # Check for currency codes
        currency_codes = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
        }
        
        for code, symbol in currency_codes.items():
            if code in text.upper():
                return symbol
        
        return None
    
    @staticmethod
    def _extract_price_value(text: str) -> Optional[float]:
        """
        Extract numeric price value from text.
        
        Args:
            text: Input text (should be cleaned)
            
        Returns:
            Price value or None
        """
        # Remove currency symbols and letters
        text_cleaned = re.sub(r'[^\d.,]', '', text)
        
        # Try different decimal separators
        patterns = [
            r'(\d+[.,]\d{2})',  # Exact decimal: 19.99 or 19,99
            r'(\d+[.,]\d)',      # One decimal: 19.9
            r'(\d+)',            # Integer: 19
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_cleaned)
            if match:
                price_str = match.group(1)
                # Normalize: replace comma with dot, then convert
                price_str = price_str.replace(',', '.')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        
        return None
    
    def _validate_price(self, price: float) -> bool:
        """
        Validate price is within acceptable range.
        
        Args:
            price: Price value
            
        Returns:
            True if valid, False otherwise
        """
        return self.min_price <= price <= self.max_price


class PriceValidator:
    """Validate prices and detect anomalies."""
    
    def __init__(self, 
                 expected_price: Optional[float] = None,
                 tolerance: float = 0.05):  # 5% tolerance
        """
        Initialize price validator.
        
        Args:
            expected_price: Expected price for comparison
            tolerance: Price difference tolerance (as decimal, e.g., 0.05 for 5%)
        """
        self.expected_price = expected_price
        self.tolerance = tolerance
        logger.info(f"PriceValidator initialized: expected={expected_price}, tolerance={tolerance*100}%")
    
    def validate(self, price_data: PriceData) -> Dict:
        """
        Validate price data and check for anomalies.
        
        Args:
            price_data: PriceData object
            
        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': price_data.is_valid,
            'price_value': price_data.price_value,
            'confidence': price_data.confidence,
            'errors': [],
            'warnings': [],
            'price_mismatch': False,
            'price_difference': None
        }
        
        if not price_data.is_valid:
            result['errors'].append(price_data.error_message)
            return result
        
        # Check OCR confidence
        if price_data.confidence < 0.7:
            result['warnings'].append(f"Low OCR confidence: {price_data.confidence:.2%}")
        
        # Check against expected price
        if self.expected_price is not None and price_data.price_value is not None:
            difference = abs(price_data.price_value - self.expected_price) / self.expected_price
            result['price_difference'] = difference
            
            if difference > self.tolerance:
                result['price_mismatch'] = True
                result['errors'].append(
                    f"Price mismatch: {price_data.formatted_price()} "
                    f"vs expected {self.expected_price} (diff: {difference:.2%})"
                )
        
        return result
