#!/usr/bin/env python3
"""
Test script for the enhanced medical image analysis system
Tests all three datasets and visualization features
"""

import os
import sys
import logging
from pathlib import Path
import random
from PIL import Image
import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_dataset_structure():
    """Test if dataset is properly structured"""
    logger.info("🔍 Testing dataset structure...")
    
    datasets = {
        'brain_tumor': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'chest_xray': ['NORMAL', 'PNEUMONIA'],
        'bone_fracture': ['fractured', 'not fractured']
    }
    
    all_good = True
    
    for dataset_name, classes in datasets.items():
        logger.info(f"Testing {dataset_name} dataset...")
        
        for split in ['train', 'val', 'test']:
            for class_name in classes:
                path = Path(f"dataset/{split}/{dataset_name}/{class_name}")
                if path.exists():
                    image_count = len(list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png")))
                    logger.info(f"  ✅ {split}/{dataset_name}/{class_name}: {image_count} images")
                else:
                    logger.error(f"  ❌ Missing: {path}")
                    all_good = False
    
    return all_good

def test_model_loading():
    """Test if model can be loaded"""
    logger.info("🔍 Testing model loading...")
    
    try:
        # Import the enhanced app
        from medical_app_enhanced import load_model, device
        
        # Load model
        load_model()
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"✅ Using device: {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_prediction_pipeline():
    """Test the prediction pipeline with sample images"""
    logger.info("🔍 Testing prediction pipeline...")
    
    try:
        from medical_app_enhanced import predict_medical_image, safe_image_load
        
        # Find sample images from each dataset
        sample_images = []
        
        # Brain tumor sample
        brain_paths = list(Path("dataset/test/brain_tumor").rglob("*.jpg"))[:1]
        if brain_paths:
            sample_images.append(("Brain Tumor", brain_paths[0]))
        
        # Chest X-ray sample
        chest_paths = list(Path("dataset/test/chest_xray").rglob("*.jpeg"))[:1]
        if chest_paths:
            sample_images.append(("Chest X-ray", chest_paths[0]))
        
        # Bone fracture sample
        bone_paths = list(Path("dataset/test/bone_fracture").rglob("*.jpg"))[:1]
        if bone_paths:
            sample_images.append(("Bone Fracture", bone_paths[0]))
        
        if not sample_images:
            logger.warning("⚠️  No test images found")
            return False
        
        # Test predictions
        for dataset_type, image_path in sample_images:
            logger.info(f"Testing {dataset_type} prediction...")
            
            try:
                # Load and test image
                image = safe_image_load(str(image_path))
                
                # Make prediction
                diagnosis, remedy, report, viz, highlighted = predict_medical_image(
                    image, threshold=0.1, modality="Auto"
                )
                
                logger.info(f"  ✅ {dataset_type}: {diagnosis[:50]}...")
                
                # Check if highlighted image was generated
                if highlighted is not None:
                    logger.info(f"  ✅ Grad-CAM visualization generated")
                else:
                    logger.warning(f"  ⚠️  Grad-CAM visualization not generated")
                
            except Exception as e:
                logger.error(f"  ❌ {dataset_type} prediction failed: {e}")
                return False
        
        logger.info("✅ Prediction pipeline test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction pipeline test failed: {e}")
        return False

def test_gradcam_visualization():
    """Test Grad-CAM visualization specifically"""
    logger.info("🔍 Testing Grad-CAM visualization...")
    
    try:
        from medical_app_enhanced import create_gradcam_visualization, load_model, device
        from torchvision import transforms
        
        # Load model
        load_model()
        
        # Find a test image
        test_images = list(Path("dataset/test").rglob("*.jpg")) + list(Path("dataset/test").rglob("*.jpeg"))
        if not test_images:
            logger.warning("⚠️  No test images found for Grad-CAM test")
            return False
        
        # Test with first available image
        test_image_path = test_images[0]
        logger.info(f"Testing Grad-CAM with: {test_image_path}")
        
        # Load image
        image = Image.open(test_image_path).convert('RGB')
        
        # Test Grad-CAM
        highlighted = create_gradcam_visualization(image, None, 0, "test_class")
        
        if highlighted is not None and highlighted.size == image.size:
            logger.info("✅ Grad-CAM visualization working correctly")
            return True
        else:
            logger.error("❌ Grad-CAM visualization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Grad-CAM test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Enhanced Medical Image Analysis System Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Grad-CAM Visualization", test_gradcam_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        logger.info("🚀 You can now run: python launch_enhanced_app.py")
    else:
        logger.error("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

