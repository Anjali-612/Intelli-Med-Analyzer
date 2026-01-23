#!/usr/bin/env python3
"""
Test script to verify that predictions are working correctly
"""

import os
import sys
import logging
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_predictions():
    """Test predictions on sample images from each class"""
    logger.info("🧪 Testing predictions on sample images...")
    
    try:
        # Import the enhanced app
        from medical_app_enhanced import load_model_safe, CLASS_NAMES, preprocess_for_model
        
        # Load model
        logger.info("Loading model...")
        model = load_model_safe("model/best_model.pth", len(CLASS_NAMES))
        logger.info(f"Model loaded successfully with {len(CLASS_NAMES)} classes")
        logger.info(f"Class names: {CLASS_NAMES}")
        
        # Test images from each class
        test_images = []
        
        # Brain tumor classes
        brain_classes = ["glioma", "meningioma", "notumor", "pituitary"]
        for brain_class in brain_classes:
            brain_paths = list(Path("dataset/test/brain_tumor").glob(f"{brain_class}/*.jpg"))
            if brain_paths:
                test_images.append((f"brain_tumor_{brain_class}", brain_paths[0]))
        
        # Chest X-ray classes
        chest_paths_normal = list(Path("dataset/test/chest_xray/NORMAL").glob("*.jpeg"))
        chest_paths_pneumonia = list(Path("dataset/test/chest_xray/PNEUMONIA").glob("*.jpeg"))
        if chest_paths_normal:
            test_images.append(("chest_xray_NORMAL", chest_paths_normal[0]))
        if chest_paths_pneumonia:
            test_images.append(("chest_xray_PNEUMONIA", chest_paths_pneumonia[0]))
        
        # Bone fracture classes
        bone_paths_fractured = list(Path("dataset/test/bone_fracture/fractured").glob("*.jpg"))
        bone_paths_not_fractured = list(Path("dataset/test/bone_fracture/not fractured").glob("*.jpg"))
        if bone_paths_fractured:
            test_images.append(("bone_fracture_fractured", bone_paths_fractured[0]))
        if bone_paths_not_fractured:
            test_images.append(("bone_fracture_not fractured", bone_paths_not_fractured[0]))
        
        if not test_images:
            logger.error("No test images found!")
            return False
        
        logger.info(f"Found {len(test_images)} test images")
        
        # Test each image
        correct_predictions = 0
        total_predictions = 0
        
        for expected_class, image_path in test_images:
            logger.info(f"\n🔍 Testing: {expected_class}")
            logger.info(f"Image: {image_path}")
            
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                input_tensor = preprocess_for_model(image)
                
                # Make prediction
                with torch.no_grad():
                    model_output = model(input_tensor)
                    if isinstance(model_output, tuple):
                        logits, attention_maps = model_output
                    else:
                        logits = model_output
                    
                    # Get probabilities
                    probs = F.softmax(logits, dim=1)
                    confidence_val, pred_idx = torch.max(probs, 1)
                    pred_idx = int(pred_idx[0].cpu().numpy())
                    confidence_val = float(confidence_val[0].cpu().numpy())
                
                predicted_class = CLASS_NAMES[pred_idx]
                
                # Check if prediction is correct
                is_correct = predicted_class == expected_class
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Display results
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                logger.info(f"Expected: {expected_class}")
                logger.info(f"Predicted: {predicted_class}")
                logger.info(f"Confidence: {confidence_val:.3f}")
                logger.info(f"Status: {status}")
                
                # Show top 3 predictions
                top3_probs, top3_indices = torch.topk(probs[0], 3)
                logger.info("Top 3 predictions:")
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    logger.info(f"  {i+1}. {CLASS_NAMES[idx]}: {prob:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        # Summary
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"Correct predictions: {correct_predictions}/{total_predictions}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        
        if accuracy > 0.5:
            logger.info("✅ Predictions are working correctly!")
            return True
        else:
            logger.warning("⚠️  Low accuracy - predictions may need improvement")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_predictions()
    sys.exit(0 if success else 1)
