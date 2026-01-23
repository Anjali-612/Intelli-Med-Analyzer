#!/usr/bin/env python3
"""
Direct Medical App Runner - Minimal launcher to avoid Windows path issues
"""

import os
import sys

# Set environment variables
os.environ['GRADIO_SERVER_PORT'] = '7860'

print("🩺 Medical Image Analysis App")
print("Starting server...")
print("If browser doesn't open automatically, go to: http://127.0.0.1:7860")
print("Press Ctrl+C to stop")

try:
    # Import and run the app directly
    from medical_app_enhanced import main
    main()
except KeyboardInterrupt:
    print("\n👋 App stopped")
except Exception as e:
    print(f"❌ Error: {e}")
    input("Press Enter to exit...")


















