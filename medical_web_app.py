import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np
import gradio as gr
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "medical_model.pth"


# ---------------------------------------------------
# LOAD CHECKPOINT (your exact saved structure)
# ---------------------------------------------------
def load_model():
    print("Loading checkpoint:", MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = ckpt["class_names"]  # ORIGINAL TRAIN ORDER
    display_names = ckpt["display_names"]
    norm = ckpt["norm"]

    # Build EfficientNet-B0 EXACTLY as trained
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    # load weights
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    return model, class_names, display_names, norm


model, class_names, display_names, norm_vals = load_model()


# ---------------------------------------------------
# Preprocessing (uses SAME mean/std from checkpoint)
# ---------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_vals["mean"], norm_vals["std"])
])


# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------
def predict_image(img):

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Preprocess
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))

    # Map → correct label
    class_name = class_names[pred_idx]        # dataset folder name
    display_name = display_names[pred_idx]    # mapped UI label

    confidence = float(probs[pred_idx]) * 100

    return f"Prediction: {display_name}\nConfidence: {confidence:.2f}%"


# ---------------------------------------------------
# GRAD-CAM++ IMPLEMENTATION FOR EFFICIENTNET
# ---------------------------------------------------
class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook last conv layer of EfficientNet
        target_layer = model.features[-1][0].block[2][0]  # (correct for B0)

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        pred_class = output.argmax()
        score = output[0, pred_class]
        score.backward()

        grads = self.gradients.cpu().numpy()
        activs = self.activations.cpu().numpy()

        weights = np.mean(grads, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activs, axis=1).squeeze()

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam


gradcam = GradCAMPlusPlus(model)


# ---------------------------------------------------
# GRAD-CAM PROCESSOR
# ---------------------------------------------------
def generate_heatmap(img):

    if img.mode != "RGB":
        img = img.convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    cam = gradcam.generate(tensor)
    cam = np.uint8(255 * cam)

    cam = Image.fromarray(cam).resize(img.size, Image.BILINEAR)
    cam = cam.convert("RGB")

    # Overlay
    heat = np.array(cam).astype(float)
    base = np.array(img).astype(float)

    overlay = (0.6 * base + 0.4 * heat).astype(np.uint8)

    return Image.fromarray(overlay)


# ---------------------------------------------------
# GRADIO UI
# ---------------------------------------------------
ui = gr.Interface(
    fn=lambda img: (
        predict_image(img),
        generate_heatmap(img)
    ),
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Grad-CAM++")
    ],
    title="IntelliMed Analyzer - EfficientNet Medical Image Classifier",
    description="Upload X-ray, MRI, or Fracture image."
)

ui.launch()
