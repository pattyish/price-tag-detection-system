#!/usr/bin/env python
"""
Inference Script - Run price detection on images
Usage: python scripts/infer.py --image <path> [--output <output_dir>] [--config <config_file>]
"""

import argparse
import sys
from pathlib import Path

from src.utils.logger import PriceDetectionLogger
from src.utils.config import PriceDetectionConfig
from src.inference.pipeline import PipelineBuilder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description='Run price tag detection and OCR on images'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to directory with multiple images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8m',
        help='YOLO model name (yolov8n/s/m/l/x)'
    )
    
    parser.add_argument(
        '--ocr',
        type=str,
        default='easyocr',
        help='OCR engine (easyocr or tesseract)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    PriceDetectionLogger.setup_logging(log_level=args.log_level)
    logger = PriceDetectionLogger.get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("PRICE TAG DETECTION INFERENCE")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline
        if args.config:
            logger.info(f"Loading config from: {args.config}")
            pipeline = PipelineBuilder.create_from_config_file(args.config)
        else:
            logger.info(f"Creating pipeline with: model={args.model}, ocr={args.ocr}, device={args.device}")
            pipeline = PipelineBuilder.create_custom(
                model_name=args.model,
                ocr_engine=args.ocr,
                device=args.device,
                confidence_threshold=args.confidence
            )
        
        # Process image(s)
        if args.image:
            logger.info(f"Processing single image: {args.image}")
            image_path = Path(args.image)
            
            if not image_path.exists():
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)
            
            result = pipeline.process_image(str(image_path))
            
            # Save results
            output_file = output_dir / f"{image_path.stem}_results.json"
            pipeline.save_results(result, str(output_file))
            
            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Image: {image_path.name}")
            logger.info(f"Total detections: {result['summary']['total_detections']}")
            logger.info(f"Valid prices: {result['summary']['valid_extractions']}")
            logger.info(f"Extraction rate: {result['summary']['extraction_rate']:.1%}")
            
            if result['detections']:
                logger.info("\nDetected Prices:")
                for det in result['detections']:
                    if 'price' in det and det['price'] and det['price']['is_valid']:
                        logger.info(f"  - {det['price']['formatted_price']} (confidence: {det['price']['confidence']:.2%})")
            
            logger.info(f"\nResults saved to: {output_file}")
        
        elif args.batch:
            logger.info(f"Processing batch from directory: {args.batch}")
            batch_dir = Path(args.batch)
            
            if not batch_dir.is_dir():
                logger.error(f"Batch directory not found: {args.batch}")
                sys.exit(1)
            
            # Find all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            images = sorted([
                f for f in batch_dir.glob('*')
                if f.suffix.lower() in image_extensions
            ])
            
            if not images:
                logger.warning(f"No image files found in {args.batch}")
                sys.exit(1)
            
            logger.info(f"Found {len(images)} images to process")
            results = pipeline.process_batch([str(img) for img in images])
            
            # Save batch results
            output_file = output_dir / "batch_results.json"
            pipeline.save_results(results, str(output_file))
            
            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("BATCH RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total images: {len(results)}")
            
            total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in results)
            total_valid = sum(r.get('summary', {}).get('valid_extractions', 0) for r in results)
            
            logger.info(f"Total detections: {total_detections}")
            logger.info(f"Valid prices: {total_valid}")
            logger.info(f"Results saved to: {output_file}")
        
        else:
            parser.print_help()
            sys.exit(1)
        
        logger.info("=" * 80)
        logger.info("Inference complete!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
