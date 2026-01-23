#!/usr/bin/env python3
"""
Demo script for the Enhanced Medical Image Analysis System
Shows how to use the system programmatically
"""

import os
import sys
from pathlib import Path
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def demo_prediction():
    """Demonstrate the prediction system"""
    logger.info("🎯 Demo: Medical Image Prediction System")
    logger.info("=" * 50)
    
    try:
        # Import the enhanced app
        from medical_app_enhanced import predict_medical_image, load_model
        
        # Load the model
        logger.info("Loading model...")
        load_model()
        
        # Find sample images
        sample_images = []
        
        # Brain tumor sample
        brain_paths = list(Path("dataset/test/brain_tumor").rglob("*.jpg"))[:1]
        if brain_paths:
            sample_images.append(("Brain MRI", brain_paths[0], "Brain"))
        
        # Chest X-ray sample
        chest_paths = list(Path("dataset/test/chest_xray").rglob("*.jpeg"))[:1]
        if chest_paths:
            sample_images.append(("Chest X-ray", chest_paths[0], "Chest"))
        
        # Bone fracture sample
        bone_paths = list(Path("dataset/test/bone_fracture").rglob("*.jpg"))[:1]
        if bone_paths:
            sample_images.append(("Bone X-ray", bone_paths[0], "Bone"))
        
        if not sample_images:
            logger.error("No sample images found!")
            return False
        
        # Test each sample
        for image_type, image_path, modality in sample_images:
            logger.info(f"\n🔍 Testing {image_type}...")
            logger.info(f"Image: {image_path}")
            
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                logger.info(f"Image size: {image.size}")
                
                # Make prediction
                diagnosis, remedy, report, viz, highlighted = predict_medical_image(
                    image, threshold=0.1, modality=modality
                )
                
                # Display results
                logger.info(f"📊 Diagnosis: {diagnosis}")
                logger.info(f"💊 Treatment: {remedy[:100]}...")
                
                # Check if affected areas were highlighted
                if highlighted is not None:
                    logger.info("✅ Affected areas highlighted successfully")
                    # Save highlighted image
                    output_path = f"demo_{image_type.lower().replace(' ', '_')}_highlighted.jpg"
                    highlighted.save(output_path)
                    logger.info(f"💾 Saved highlighted image: {output_path}")
                else:
                    logger.warning("⚠️  No highlighting generated")
                
                logger.info("✅ Prediction completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_type}: {e}")
                continue
        
        logger.info("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def demo_features():
    """Demonstrate key features"""
    logger.info("\n🌟 Key Features of the Enhanced System:")
    logger.info("=" * 50)
    
    features = [
        "🎯 **Exact Image Classification**: Predicts specific conditions from medical images",
        "🔍 **Affected Area Highlighting**: Uses Grad-CAM to show which regions are important",
        "📊 **Affected/Not Affected Status**: Clear binary classification for each condition",
        "🧠 **Brain Tumor Detection**: Glioma, Meningioma, Pituitary, No Tumor",
        "🫁 **Chest X-ray Analysis**: Normal vs Pneumonia detection",
        "🦴 **Bone Fracture Detection**: Fractured vs Not Fractured",
        "📈 **Confidence Visualization**: Charts showing prediction probabilities",
        "💊 **Medical Reports**: Detailed treatment recommendations",
        "🎨 **Professional UI**: Clean, medical-grade interface"
    ]
    
    for feature in features:
        logger.info(f"  {feature}")
    
    logger.info("\n🚀 How to Use:")
    logger.info("  1. Run: python launch_enhanced_app.py")
    logger.info("  2. Upload a medical image (Brain MRI, Chest X-ray, or Bone X-ray)")
    logger.info("  3. Click 'Analyze Image'")
    logger.info("  4. View results in different tabs:")
    logger.info("     - Diagnosis: Predicted condition and confidence")
    logger.info("     - Affected Areas: Highlighted regions using Grad-CAM")
    logger.info("     - Confidence Chart: Probability distribution")
    logger.info("     - Detailed Report: Medical information and recommendations")

def main():
    """Main demo function"""
    print("🩺 Enhanced Medical Image Analysis System - Demo")
    print("=" * 60)
    
    # Show features
    demo_features()
    
    # Run prediction demo
    success = demo_prediction()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("🚀 Ready to launch the full application!")
        print("   Run: python launch_enhanced_app.py")
    else:
        print("\n❌ Demo encountered issues.")
        print("🔧 Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

