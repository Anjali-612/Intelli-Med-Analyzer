import json
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
import numpy as np
import cv2
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 1. Load EXACT class names
# ---------------------------------------------------------
with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)

with open("display_names.json") as f:
    DISPLAY_NAMES = json.load(f)


# ---------------------------------------------------------
# 2. Build SAME EfficientNet-B0 as train
# ---------------------------------------------------------
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    ckpt = torch.load("medical_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()


# ---------------------------------------------------------
# 3. Preprocess
# ---------------------------------------------------------
PRED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# ---------------------------------------------------------
# 4. Grad-CAM++
# ---------------------------------------------------------
class GradCAMpp:
    def __init__(self, model, layer_name="features.7"):
        self.model = model
        self.layer = dict([*model.named_modules()])[layer_name]

        self.activations = None
        self.gradients = None

        self.layer.register_forward_hook(self.save_activation)
        self.layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients
        acts = self.activations

        alpha = grads.pow(2) / (2 * grads.pow(2) +
                (acts * grads.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-6)
        weights = (alpha * grads).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))

        cam -= cam.min()
        cam /= cam.max() + 1e-6

        return cam


gradcam = GradCAMpp(MODEL)


# ---------------------------------------------------------
# 5. Prediction + heatmap control
# ---------------------------------------------------------
DEFECT_CLASSES = {
    "brain_tumor/glioma",
    "brain_tumor/meningioma",
    "brain_tumor/pituitary",
    "chest_xray/PNEUMONIA",
    "bone_fracture/fractured",
}

def predict(img):
    pil_img = img.convert("RGB")
    tensor_img = PRED_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = MODEL(tensor_img)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_label = DISPLAY_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())

    # --------------------------------------------
    # 🔥 ONLY generate Grad-CAM for DEFECT classes
    # --------------------------------------------
    if pred_label in DEFECT_CLASSES:
        cam = gradcam.generate(tensor_img, pred_idx)
        cam = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        img_np = cv2.cvtColor(np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    else:
        # 👉 Normal class → return original image (NO heatmap)
        overlay_rgb = np.array(pil_img.resize((224, 224)))

    # Remedy
    suggestion = REMEDIES.get(pred_label, "No remedy available.")

    return pred_label, confidence, overlay_rgb, suggestion


# ---------------------------------------------------------
# 6. Remedies
# ---------------------------------------------------------
REMEDIES = {
    "chest_xray/NORMAL": "No issues detected. Maintain healthy lifestyle.",
    "chest_xray/PNEUMONIA": "Drink warm fluids, rest, take steam inhalation. Visit doctor if severe.",
    "bone_fracture/fractured": "Immobilize the area, apply cold pack, take calcium-rich foods.",
    "bone_fracture/not fractured": "No fracture detected, but pain may need rest.",
    "brain_tumor/glioma": "Consult neurologist. Meditation, turmeric, vitamin D support recovery.",
    "brain_tumor/meningioma": "Most meningiomas are benign and slow-growing; regular MRI scans are advised to monitor changes over time.",
    "brain_tumor/notumor": "Healthy brain MRI detected.",
    "brain_tumor/pituitary": "Check hormone levels. Meditation helps with recovery."
}


# ---------------------------------------------------------
# 7. Gradio UI
# ---------------------------------------------------------
def app():
    input_img = gr.Image(type="pil", label="Upload Medical Image")
    output_label = gr.Textbox(label="Predicted Class")
    output_conf = gr.Number(label="Confidence")
    output_cam = gr.Image(label="Grad-CAM++ Heatmap")
    output_remedy = gr.Textbox(label="Suggested Remedy")

    iface = gr.Interface(
        fn=predict,
        inputs=input_img,
        outputs=[output_label, output_conf, output_cam, output_remedy],
        title="IntelliMed Analyzer – Multi-Disease Medical Image Diagnosis",
        description="Upload X-ray, MRI, or fracture image. Supports 8-class diagnosis."
    )

    iface.launch(debug=True)


if __name__ == "__main__":
    app()
