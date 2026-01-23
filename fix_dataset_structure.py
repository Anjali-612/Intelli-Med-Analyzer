#!/usr/bin/env python3
"""
Fix dataset structure by creating missing validation directories
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_missing_validation_dirs():
    """Create missing validation directories for brain tumor"""
    
    # Check if brain tumor validation directories exist
    brain_val_path = Path("dataset/val/brain_tumor")
    
    if not brain_val_path.exists():
        logger.info("Creating brain tumor validation directory...")
        brain_val_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    for subdir in subdirs:
        subdir_path = brain_val_path / subdir
        if not subdir_path.exists():
            logger.info(f"Creating {subdir} validation directory...")
            subdir_path.mkdir(parents=True, exist_ok=True)
            
            # Copy a few sample images from test set if available
            test_subdir = Path(f"dataset/test/brain_tumor/{subdir}")
            if test_subdir.exists():
                test_images = list(test_subdir.glob("*.jpg"))[:5]  # Copy first 5 images
                for img in test_images:
                    dst = subdir_path / img.name
                    shutil.copy2(img, dst)
                logger.info(f"Copied {len(test_images)} sample images to {subdir} validation")

def main():
    """Main function to fix dataset structure"""
    logger.info("🔧 Fixing dataset structure...")
    
    # Check if dataset exists
    if not Path("dataset").exists():
        logger.error("Dataset directory not found!")
        return False
    
    # Create missing validation directories
    create_missing_validation_dirs()
    
    logger.info("✅ Dataset structure fixed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Dataset structure is now complete!")
    else:
        print("❌ Failed to fix dataset structure")

