"""
Inference script to test the newest model on misclassified images.
Shows the image and the model's classification.

Example:
  python inference_on_disagreements.py \
    --model-path ./outputs/20260127_120000_binary_batch16_div5_tuned.pt \
    --disagreement-dir /dls/science/groups/i23/scripts/chris/GoniOwl_metrics/disagreements \
    --output-dir ./outputs/inference_results
"""

import os
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GoniOwl.model import load_checkpoint, predict_proba, select_device
from GoniOwl.preprocess import load_image

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, force=True)
    return logging.getLogger(__name__)

logger = setup_logging()

def run_inference(
    model_path: str,
    disagreement_dir: str,
    img_height: int | None,
    img_width: int | None,
    output_dir: str,
    crop_border: int = 0,
    device: str = "auto",
) -> None:
    """Run inference on all images in disagreement folder and display results."""
    
    logger.info("Loading model from: %s", model_path)
    model, info = load_checkpoint(model_path, select_device(device))
    img_height = img_height or info["img_height"]
    img_width = img_width or info["img_width"]
    logger.info("Model loaded successfully (input %dx%d).", img_height, img_width)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [
        f for f in Path(disagreement_dir).iterdir()
        if f.suffix in image_extensions
    ]
    
    if not image_files:
        logger.warning("No image files found in: %s", disagreement_dir)
        return
    
    logger.info("Found %d images to process.", len(image_files))
    
    results = []
    
    for idx, image_path in enumerate(sorted(image_files), 1):
        logger.info("[%d/%d] Processing: %s", idx, len(image_files), image_path.name)
        
        try:
            img = load_image(image_path, (img_height, img_width), border=crop_border)
            prediction = float(predict_proba(model, img)[0])
            if prediction > 0.85:
                class_label = "pin on"
            elif prediction < 0.15:
                class_label = "pin off"
            else:
                class_label = "undetermined"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            results.append({
                'filename': image_path.name,
                'prediction': prediction,
                'label': class_label,
                'confidence': float(confidence),
            })
            
            logger.info("  → Prediction: %s (confidence: %.2f%%)", class_label, confidence * 100)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img)
            ax.set_title(f"{image_path.name}\nPrediction: {class_label} (Confidence: {confidence:.2%})", fontsize=14, fontweight='bold')
            ax.axis('off')
            output_path = os.path.join(output_dir, f"{image_path.stem}_pred.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            
            # plt.show()
            plt.close()
            
        except Exception as e:
            logger.error("Error processing %s: %s", image_path.name, str(e))
    
    import csv
    csv_path = os.path.join(output_dir, "inference_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'prediction', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("Results saved to: %s", output_dir)
    logger.info("Summary CSV saved to: %s", csv_path)
    logger.info("Processing complete. Processed %d images.", len(results))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on misclassified images.")
    parser.add_argument("--model-path", required=True, help="Path to the trained .pt model.")
    parser.add_argument("--disagreement-dir", required=True, help="Path to folder with misclassified images.")
    parser.add_argument("--img-height", type=int, default=None, help="Image height (default: from the checkpoint).")
    parser.add_argument("--img-width", type=int, default=None, help="Image width (default: from the checkpoint).")
    parser.add_argument("--crop-border", type=int, default=0, help="Pixels to crop from each side before resizing (use 100 for raw, uncropped camera frames).")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps or cpu.")
    parser.add_argument("--output-dir", default="./outputs/inference_results", help="Output directory for results.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.isdir(args.disagreement_dir):
        raise FileNotFoundError(f"Disagreement directory not found: {args.disagreement_dir}")
    
    run_inference(
        model_path=args.model_path,
        disagreement_dir=args.disagreement_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        output_dir=args.output_dir,
        crop_border=args.crop_border,
        device=args.device,
    )