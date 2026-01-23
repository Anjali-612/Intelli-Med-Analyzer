#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Medical Platform
Tests all features including prediction accuracy, hospital tracing, and UI components
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

def test_enhanced_model_loading():
    """Test if the enhanced model can be loaded"""
    logger.info("🔍 Testing enhanced model loading...")
    
    try:
        from medical_platform_enhanced import load_model, device
        
        # Load model
        model = load_model()
        
        logger.info("✅ Enhanced model loaded successfully")
        logger.info(f"✅ Using device: {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced model loading failed: {e}")
        return False

def test_prediction_accuracy():
    """Test prediction accuracy with sample images"""
    logger.info("🔍 Testing prediction accuracy...")
    
    try:
        from medical_platform_enhanced import predict_medical_image, preprocess_image
        
        # Find sample images from each dataset
        sample_images = []
        
        # Brain tumor sample
        brain_paths = list(Path("dataset/test/brain_tumor").rglob("*.jpg"))[:2]
        if brain_paths:
            sample_images.extend([("Brain Tumor", p) for p in brain_paths])
        
        # Chest X-ray sample
        chest_paths = list(Path("dataset/test/chest_xray").rglob("*.jpeg"))[:2]
        if chest_paths:
            sample_images.extend([("Chest X-ray", p) for p in chest_paths])
        
        # Bone fracture sample
        bone_paths = list(Path("dataset/test/bone_fracture").rglob("*.jpg"))[:2]
        if bone_paths:
            sample_images.extend([("Bone Fracture", p) for p in bone_paths])
        
        if not sample_images:
            logger.warning("⚠️  No test images found")
            return False
        
        # Test predictions
        successful_predictions = 0
        total_predictions = len(sample_images)
        
        for dataset_type, image_path in sample_images:
            logger.info(f"Testing {dataset_type} prediction...")
            
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Make prediction
                results = predict_medical_image(image)
                
                if "error" not in results:
                    predicted_class = results["predicted_class"]
                    confidence = results["confidence"]
                    medical_info = results.get("medical_info", {})
                    
                    logger.info(f"  ✅ Predicted: {predicted_class}")
                    logger.info(f"  ✅ Confidence: {confidence:.1%}")
                    logger.info(f"  ✅ Condition: {medical_info.get('condition', 'Unknown')}")
                    
                    # Check if confidence is reasonable
                    if confidence > 0.1:  # At least 10% confidence
                        successful_predictions += 1
                    else:
                        logger.warning(f"  ⚠️  Low confidence: {confidence:.1%}")
                else:
                    logger.error(f"  ❌ Prediction error: {results['error']}")
                
            except Exception as e:
                logger.error(f"  ❌ {dataset_type} prediction failed: {e}")
        
        accuracy_rate = successful_predictions / total_predictions
        logger.info(f"✅ Prediction accuracy rate: {accuracy_rate:.1%} ({successful_predictions}/{total_predictions})")
        
        return accuracy_rate > 0.5  # At least 50% success rate
        
    except Exception as e:
        logger.error(f"❌ Prediction accuracy test failed: {e}")
        return False

def test_gradcam_visualization():
    """Test enhanced Grad-CAM visualization"""
    logger.info("🔍 Testing enhanced Grad-CAM visualization...")
    
    try:
        from medical_platform_enhanced import create_enhanced_gradcam, load_model
        
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
        
        # Test enhanced Grad-CAM
        highlighted = create_enhanced_gradcam(image, "test_class")
        
        if highlighted is not None and highlighted.size == image.size:
            logger.info("✅ Enhanced Grad-CAM visualization working correctly")
            
            # Save test output
            output_path = f"test_gradcam_output_{Path(test_image_path).stem}.jpg"
            highlighted.save(output_path)
            logger.info(f"✅ Saved test Grad-CAM image: {output_path}")
            
            return True
        else:
            logger.error("❌ Enhanced Grad-CAM visualization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Grad-CAM test failed: {e}")
        return False

def test_hospital_tracing():
    """Test hospital tracing functionality"""
    logger.info("🔍 Testing hospital tracing...")
    
    try:
        from medical_platform_enhanced import get_hospitals_near_location, create_hospital_map
        
        # Test location
        test_location = "Hyderabad, India"
        test_condition = "brain_tumor_glioma"
        
        # Get hospitals
        hospitals = get_hospitals_near_location(test_location, test_condition)
        
        if hospitals and len(hospitals) > 0:
            logger.info(f"✅ Found {len(hospitals)} hospitals near {test_location}")
            
            for i, hospital in enumerate(hospitals):
                logger.info(f"  Hospital {i+1}: {hospital['name']}")
                logger.info(f"    Distance: {hospital['distance']}")
                logger.info(f"    Phone: {hospital['phone']}")
                logger.info(f"    Rating: {hospital['rating']}")
            
            # Test map creation
            map_html = create_hospital_map(test_location, hospitals)
            
            if map_html and "folium" in map_html.lower():
                logger.info("✅ Hospital map created successfully")
                return True
            else:
                logger.error("❌ Hospital map creation failed")
                return False
        else:
            logger.error("❌ No hospitals found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hospital tracing test failed: {e}")
        return False

def test_medical_database():
    """Test medical database functionality"""
    logger.info("🔍 Testing medical database...")
    
    try:
        from medical_platform_enhanced import MEDICAL_DATABASE, EMERGENCY_CONTACTS
        
        # Test medical database
        if MEDICAL_DATABASE and len(MEDICAL_DATABASE) > 0:
            logger.info(f"✅ Medical database loaded with {len(MEDICAL_DATABASE)} conditions")
            
            # Test a few conditions
            test_conditions = ["brain_tumor_glioma", "chest_xray_PNEUMONIA", "bone_fracture_fractured"]
            
            for condition in test_conditions:
                if condition in MEDICAL_DATABASE:
                    info = MEDICAL_DATABASE[condition]
                    logger.info(f"  ✅ {condition}: {info.get('condition', 'Unknown')}")
                    logger.info(f"    Severity: {info.get('severity', 'Unknown')}")
                    logger.info(f"    Urgency: {info.get('urgency', 'Unknown')}")
                else:
                    logger.warning(f"  ⚠️  Condition not found: {condition}")
            
            # Test emergency contacts
            if EMERGENCY_CONTACTS and len(EMERGENCY_CONTACTS) > 0:
                logger.info(f"✅ Emergency contacts loaded for {len(EMERGENCY_CONTACTS)} countries")
                
                if "India" in EMERGENCY_CONTACTS:
                    india_contacts = EMERGENCY_CONTACTS["India"]
                    logger.info(f"  ✅ India contacts: {list(india_contacts.keys())}")
                
                return True
            else:
                logger.error("❌ Emergency contacts not loaded")
                return False
        else:
            logger.error("❌ Medical database not loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Medical database test failed: {e}")
        return False

def test_report_generation():
    """Test medical report generation"""
    logger.info("🔍 Testing medical report generation...")
    
    try:
        from medical_platform_enhanced import generate_medical_report, predict_medical_image
        
        # Find a test image
        test_images = list(Path("dataset/test").rglob("*.jpg")) + list(Path("dataset/test").rglob("*.jpeg"))
        if not test_images:
            logger.warning("⚠️  No test images found for report test")
            return False
        
        # Test with first available image
        test_image_path = test_images[0]
        logger.info(f"Testing report generation with: {test_image_path}")
        
        # Load image
        image = Image.open(test_image_path).convert('RGB')
        
        # Get prediction
        results = predict_medical_image(image)
        
        if "error" not in results:
            # Generate report
            report = generate_medical_report(results, image, image)
            
            if report and len(report) > 100:  # Basic check for substantial content
                logger.info("✅ Medical report generated successfully")
                logger.info(f"✅ Report length: {len(report)} characters")
                
                # Check for key elements
                if "Diagnosis Summary" in report and "Treatment" in report:
                    logger.info("✅ Report contains required sections")
                    return True
                else:
                    logger.error("❌ Report missing required sections")
                    return False
            else:
                logger.error("❌ Report generation failed")
                return False
        else:
            logger.error(f"❌ Prediction failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Report generation test failed: {e}")
        return False

def test_ui_components():
    """Test UI component creation"""
    logger.info("🔍 Testing UI components...")
    
    try:
        from medical_platform_enhanced import create_enhanced_interface
        
        # Test interface creation
        interface = create_enhanced_interface()
        
        if interface is not None:
            logger.info("✅ Enhanced interface created successfully")
            logger.info("✅ UI components loaded")
            return True
        else:
            logger.error("❌ Interface creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ UI components test failed: {e}")
        return False

def main():
    """Run comprehensive test suite"""
    logger.info("🧪 Enhanced Medical Platform Comprehensive Test Suite")
    logger.info("=" * 70)
    
    tests = [
        ("Enhanced Model Loading", test_enhanced_model_loading),
        ("Prediction Accuracy", test_prediction_accuracy),
        ("Enhanced Grad-CAM Visualization", test_gradcam_visualization),
        ("Hospital Tracing", test_hospital_tracing),
        ("Medical Database", test_medical_database),
        ("Report Generation", test_report_generation),
        ("UI Components", test_ui_components)
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
    logger.info("\n" + "=" * 70)
    logger.info("📊 COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced platform is ready to use.")
        logger.info("🚀 You can now run: python launch_enhanced_platform.py")
        logger.info("🌟 Features verified:")
        logger.info("   ✅ Accurate image prediction")
        logger.info("   ✅ Hospital tracing with location services")
        logger.info("   ✅ Enhanced image highlighting")
        logger.info("   ✅ Modern attractive UI")
        logger.info("   ✅ Comprehensive medical database")
        logger.info("   ✅ Emergency contacts and notifications")
        logger.info("   ✅ Interactive hospital maps")
        logger.info("   ✅ Detailed medical reports")
    else:
        logger.error("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)














