"""Setup configuration for Price Detection System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="price-detection-system",
    version="1.0.0",
    author="Retail AI Team",
    description="Production-ready system for detecting and extracting price information from retail shelf images using YOLO and OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/price-detection-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Retail",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "easyocr>=1.6.0",
        "pytesseract>=0.3.10",
        "Pillow>=9.0.0",
        "scikit-image>=0.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.0.0",
        "python-json-logger>=2.0.4",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pydantic>=1.10.0",
        "streamlit>=1.36.0",
    ],
    extras_require={
        "all": [
            "albumentations>=1.3.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "price-detect-infer=scripts.infer:main",
        ],
    },
)
