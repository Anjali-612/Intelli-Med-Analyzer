#!/usr/bin/env python3
"""
Simple Medical App Runner - Bypasses training to avoid Windows path issues
"""

import os
import sys
import socket
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_free_port(start_port=7860, max_port=8000):
    """Find a free port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def check_dependencies():
    """Check if required packages are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'gradio', 
        'PIL', 'numpy', 'matplotlib', 'tqdm', 'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.info(f"❌ {package}")
    
    if missing_packages:
        logger.info(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        logger.info("📦 Please install missing packages:")
        logger.info("pip install torch torchvision gradio Pillow numpy matplotlib tqdm opencv-python")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🩺 Simple Medical Image Analysis App")
    print("=" * 50)
    print("Features:")
    print("• Predicts and classifies medical images from 3 datasets")
    print("• Highlights affected areas using Grad-CAM visualization")
    print("• Shows 'affected' vs 'not affected' status")
    print("• Uses dummy model to avoid Windows path issues")
    print("=" * 50)
    
    try:
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        # Find free port
        port = find_free_port()
        if port is None:
            logger.error("❌ No free port found in range 7860-8000")
            return False
        
        logger.info(f"🚀 Starting app on port {port}")
        
        # Set environment variable for port
        os.environ['GRADIO_SERVER_PORT'] = str(port)
        
        # Import and run the enhanced medical app
        logger.info("📱 Loading medical app...")
        from medical_app_enhanced import main as app_main
        logger.info("✅ App loaded successfully, starting server...")
        app_main()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
        return True
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)


















