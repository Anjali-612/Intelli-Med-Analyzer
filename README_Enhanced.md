# Enhanced Medical Image Analysis System

## 🩺 Overview

This enhanced medical image analysis system can predict and classify medical images from three datasets with **highlighted affected areas** using advanced AI techniques:

- **Brain MRI Scans** (Glioma, Meningioma, Pituitary, No Tumor)
- **Chest X-rays** (Normal, Pneumonia)
- **Bone X-rays** (Fractured, Not Fractured)

## ✨ Key Features

### 🎯 **Exact Image Classification**
- Predicts the exact condition from uploaded medical images
- Shows confidence scores for all possible conditions
- Supports auto-detection or manual selection of image type

### 🔍 **Affected Area Highlighting**
- **Grad-CAM visualization** highlights the exact areas the AI focuses on
- Shows which regions are most important for the diagnosis
- Color-coded heat maps indicate affected vs normal areas

### 📊 **Comprehensive Analysis**
- **Affected/Not Affected** status for each condition
- Detailed medical reports with treatment recommendations
- Confidence charts and probability distributions
- Emergency contact information

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements_enhanced.txt
```

### 2. **Test the System**
```bash
python test_enhanced_system.py
```

### 3. **Launch the Application**
```bash
python launch_enhanced_app.py
```

The app will automatically:
- Check your dataset structure
- Train a new model if needed
- Start the web interface
- Open in your browser at `http://127.0.0.1:7860`

## 📁 Dataset Structure

Your dataset should be organized as follows:
```
dataset/
├── train/
│   ├── brain_tumor/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   ├── chest_xray/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── bone_fracture/
│       ├── fractured/
│       └── not fractured/
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

## 🎮 How to Use

### **Step 1: Upload Image**
- Click "Upload Medical Image" and select your X-ray/MRI/CT scan
- Choose image type (Auto, Brain, Chest, or Bone) or let the system auto-detect

### **Step 2: Analyze**
- Click "🔍 Analyze Image" to start the analysis
- The system will process the image and generate results

### **Step 3: View Results**
- **Diagnosis Tab**: See the predicted condition and confidence
- **Affected Areas Tab**: View highlighted regions using Grad-CAM
- **Confidence Chart**: See probability distribution across all conditions
- **Detailed Report**: Get comprehensive medical information

## 🔬 Technical Details

### **Enhanced Model Architecture**
- **ResNet18 backbone** with attention mechanisms
- **Focal Loss** for handling class imbalance
- **Advanced data augmentation** for better generalization
- **Grad-CAM integration** for explainable AI

### **Grad-CAM Visualization**
- Highlights the most important regions for diagnosis
- Uses gradient-weighted class activation mapping
- Shows exactly where the AI "looks" when making predictions
- Color-coded: Red = High importance, Blue = Low importance

### **Classification Logic**
```python
# Brain Tumor Detection
if predicted_class in ["glioma", "meningioma", "pituitary"]:
    status = "AFFECTED - Brain Tumor Detected"
else:
    status = "NOT AFFECTED - No Brain Tumor"

# Chest X-ray Analysis  
if predicted_class == "PNEUMONIA":
    status = "AFFECTED - Pneumonia Detected"
else:
    status = "NOT AFFECTED - Normal Chest"

# Bone Fracture Detection
if predicted_class == "fractured":
    status = "AFFECTED - Bone Fracture Detected"
else:
    status = "NOT AFFECTED - No Fracture"
```

## 📊 Example Outputs

### **Brain MRI Analysis**
- **Input**: Brain MRI scan
- **Output**: "🧠 Brain Tumor Detected (Glioma) - 94.2% confidence"
- **Highlighted Areas**: Red regions showing tumor locations
- **Status**: **AFFECTED**

### **Chest X-ray Analysis**
- **Input**: Chest X-ray image
- **Output**: "🫁 No Pneumonia Detected - 87.5% confidence"
- **Highlighted Areas**: Blue regions showing normal lung tissue
- **Status**: **NOT AFFECTED**

### **Bone X-ray Analysis**
- **Input**: Bone X-ray image
- **Output**: "🦴 Bone Fracture Detected - 91.8% confidence"
- **Highlighted Areas**: Red regions showing fracture lines
- **Status**: **AFFECTED**

## 🛠️ Advanced Usage

### **Retrain the Model**
```bash
python train_enhanced.py
```

### **Custom Confidence Threshold**
- Adjust the confidence threshold slider (0.1 to 0.9)
- Lower values show more possibilities
- Higher values require more confidence

### **Batch Processing**
You can modify the code to process multiple images at once by iterating through your dataset folders.

## ⚠️ Important Notes

### **Medical Disclaimer**
- This tool is for **educational and research purposes only**
- **Always consult qualified healthcare professionals** for medical diagnosis
- The AI predictions should not replace professional medical advice

### **System Requirements**
- **Python 3.8+**
- **8GB+ RAM** recommended
- **GPU support** (CUDA) for faster processing (optional)
- **Windows 10/11** (tested on Windows)

### **Performance Tips**
- Use images with good contrast and resolution
- Ensure proper lighting in X-ray images
- For best results, use images similar to the training data

## 🔧 Troubleshooting

### **Common Issues**

1. **"Model files not found"**
   - Run `python train_enhanced.py` to train a new model
   - Ensure your dataset is properly organized

2. **"No free port found"**
   - Close other applications using ports 7860-8000
   - Restart your computer if needed

3. **"Dataset path not found"**
   - Check that your dataset folder is in the correct location
   - Ensure all subfolders exist as shown in the structure above

4. **"Grad-CAM visualization not working"**
   - This is normal for some edge cases
   - The prediction will still work, just without highlighting

### **Getting Help**
- Check the console output for detailed error messages
- Run `python test_enhanced_system.py` to diagnose issues
- Ensure all dependencies are installed correctly

## 🎯 Success Metrics

When working correctly, you should see:
- ✅ **Exact predictions** for each medical condition
- ✅ **Highlighted affected areas** in red/orange colors
- ✅ **Clear "AFFECTED" or "NOT AFFECTED" status**
- ✅ **High confidence scores** (>80% for good images)
- ✅ **Detailed medical reports** with treatment recommendations

## 🚀 Next Steps

1. **Test with your own images** to see the system in action
2. **Adjust confidence thresholds** based on your needs
3. **Retrain the model** with additional data if available
4. **Integrate with your medical workflow** as needed

---

**Ready to analyze medical images with AI-powered precision and visual explanations!** 🩺✨

