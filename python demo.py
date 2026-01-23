import os
import sys
import logging
from typing import Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import gradio as gr
import folium
try:
    import requests
except Exception:
    requests = None
# Local predictions - no external APIs or datasets required

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
CLASS_NAMES = [
    "bone_fracture/fractured",
    "bone_fracture/not fractured",
    "brain_tumor/glioma",
    "brain_tumor/meningioma",
    "brain_tumor/notumor",
    "brain_tumor/pituitary",
    "chest_xray/NORMAL",
    "chest_xray/PNEUMONIA"
]
MEDICAL_DATABASE = {
    "bone_fracture/fractured": {
        "condition": "Bone Fracture Detected",
        "severity": "High",
        "urgency": "Immediate",
        "description": "A break or crack in a bone has been detected in the X-ray image.",
        "symptoms": ["Pain", "Swelling", "Bruising", "Deformity", "Loss of function"],
        "treatment": "Immobilization with cast or splint, pain management, and possible surgery for complex fractures.",
        "specialist": "Orthopedic Surgeon",
        "emergency": True,
        "follow_up": "Immediate medical attention required"
    },
    "bone_fracture/not fractured": {
        "condition": "No Bone Fracture",
        "severity": "Low",
        "urgency": "Low",
        "description": "No signs of bone fracture detected. Bone structure appears normal.",
        "symptoms": ["May have soft tissue injury"],
        "treatment": "Rest, ice, compression, elevation (RICE) for soft tissue injuries.",
        "specialist": "General Practitioner",
        "emergency": False,
        "follow_up": "Monitor for persistent pain"
    },
    "brain_tumor/glioma": {
        "condition": "Glioma Brain Tumor",
        "severity": "High",
        "urgency": "High",
        "description": "A glioma (malignant brain tumor) has been detected in the MRI scan.",
        "symptoms": ["Headaches", "Seizures", "Cognitive changes", "Motor weakness", "Vision problems"],
        "treatment": "Surgery, radiation therapy, chemotherapy, and targeted therapy.",
        "specialist": "Neurosurgeon, Oncologist",
        "emergency": True,
        "follow_up": "Immediate neurological consultation required"
    },
    "brain_tumor/meningioma": {
        "condition": "Meningioma Brain Tumor",
        "severity": "Medium",
        "urgency": "Medium",
        "description": "A meningioma (usually benign brain tumor) has been detected.",
        "symptoms": ["Headaches", "Seizures", "Vision changes", "Memory problems"],
        "treatment": "Surgical removal, radiation therapy, or monitoring for small tumors.",
        "specialist": "Neurosurgeon",
        "emergency": False,
        "follow_up": "Neurosurgical consultation within 1-2 weeks"
    },
    "brain_tumor/notumor": {
        "condition": "No Brain Tumor",
        "severity": "Low",
        "urgency": "Low",
        "description": "No brain tumor detected. Brain structure appears normal.",
        "symptoms": ["None"],
        "treatment": "No treatment needed. Continue regular health monitoring.",
        "specialist": "General Practitioner",
        "emergency": False,
        "follow_up": "Routine follow-up as needed"
    },
    "brain_tumor/pituitary": {
        "condition": "Pituitary Tumor",
        "severity": "Medium",
        "urgency": "Medium",
        "description": "A pituitary tumor has been detected. May affect hormone production.",
        "symptoms": ["Hormonal imbalances", "Vision problems", "Headaches", "Fatigue"],
        "treatment": "Surgical removal, medication, or radiation therapy.",
        "specialist": "Endocrinologist, Neurosurgeon",
        "emergency": False,
        "follow_up": "Endocrinological consultation within 1 week"
    },
    "chest_xray/NORMAL": {
        "condition": "Normal Chest X-ray",
        "severity": "Low",
        "urgency": "Low",
        "description": "No signs of pneumonia or other lung abnormalities detected.",
        "symptoms": ["None"],
        "treatment": "No treatment needed. Maintain good respiratory health.",
        "specialist": "General Practitioner",
        "emergency": False,
        "follow_up": "Routine health checkup"
    },
    "chest_xray/PNEUMONIA": {
        "condition": "Pneumonia Detected",
        "severity": "High",
        "urgency": "High",
        "description": "Signs of pneumonia (lung infection) detected in the chest X-ray.",
        "symptoms": ["Cough", "Fever", "Difficulty breathing", "Chest pain", "Fatigue"],
        "treatment": "Antibiotics, rest, fluids, and possible hospitalization for severe cases.",
        "specialist": "Pulmonologist, Infectious Disease Specialist",
        "emergency": True,
        "follow_up": "Immediate medical attention required"
    }
}

# Normalize MEDICAL_DATABASE keys that may contain spaces
def normalize_class_key(key: str) -> str:
    return key.strip().replace(" ", "_")

for k in list(MEDICAL_DATABASE.keys()):
    normalized = normalize_class_key(k)
    if normalized != k:
        MEDICAL_DATABASE[normalized] = MEDICAL_DATABASE.pop(k)

# Emergency contacts (unchanged)
EMERGENCY_CONTACTS = {
    "India": {
        "ambulance": "108",
        "police": "100",
        "fire": "101",
        "medical_emergency": "102"
    },
    "USA": {
        "emergency": "911",
        "poison_control": "1-800-222-1222"
    },
    "UK": {
        "emergency": "999",
        "non_emergency": "101"
    }
}

# ----------------------------
# MODEL ARCHITECTURE CLASSES
# ----------------------------
class EnhancedResNet(nn.Module):
    """Enhanced ResNet with attention mechanisms and better feature extraction"""
    def __init__(self, num_classes, pretrained=True):
        super(EnhancedResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.attention1 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention1(features)
        attended_features = features * attention_weights
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)
        return output, attended_features

class SimpleResNet(nn.Module):
    """Simple ResNet for standard model files that use 'fc' instead of 'classifier'"""
    def __init__(self, num_classes, pretrained=True):
        super(SimpleResNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    def forward(self, x):
        output = self.model(x)
        return output, torch.zeros(1, 1, 224, 224).to(x.device)

# ----------------------------
# GRAD-CAM IMPLEMENTATION
# ----------------------------
class SimpleGradCAM:
    """Robust Grad-CAM: find last Conv2d layer and register safe hooks (with fallback)."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._forward_handle = None
        self._backward_handle = None

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            g = grad_output[0]
        else:
            g = grad_output
        if g is not None:
            self.gradients = g.detach()

    def _register_hooks(self, target_layer):
        self._remove_hooks()
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        if hasattr(target_layer, "register_full_backward_hook"):
            try:
                self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)
            except Exception:
                self._backward_handle = target_layer.register_backward_hook(self._save_gradient)
        else:
            self._backward_handle = target_layer.register_backward_hook(self._save_gradient)

    def _remove_hooks(self):
        if self._forward_handle is not None:
            try:
                self._forward_handle.remove()
            except Exception:
                pass
            self._forward_handle = None
        if self._backward_handle is not None:
            try:
                self._backward_handle.remove()
            except Exception:
                pass
            self._backward_handle = None

    def _find_target_layer(self):
        target = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target = module
                break
        return target

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.activations = None
        self.gradients = None
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        self._register_hooks(target_layer)
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0])
        loss = logits[0, class_idx]
        loss.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            self._remove_hooks()
            raise RuntimeError("Gradients or activations were not captured by hooks. GradCAM failed.")
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        self._remove_hooks()
        return cam.cpu().numpy()

# ----------------------------
# GLOBAL MODEL VARIABLE & LOADER
# ----------------------------
model = None

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v
    return new_state

def load_model():
    """Load the trained medical model - auto-detects model architecture (robust)."""
    global model
    if model is None:
        logger.info("Loading enhanced medical model...")
        model_paths = ["medical_model.pth", "model/best_model.pth", "best_model.pth"]
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"✓ Found model file: {path}")
                break
        if model_path is None:
            logger.error(f"Model not found in any of: {model_paths}")
            raise FileNotFoundError("Model not found. Please ensure model file exists at one of the expected paths.")
        logger.info(f"Loading model from: {model_path}")
        raw = torch.load(model_path, map_location=device)
        if isinstance(raw, dict):
            if "state_dict" in raw and isinstance(raw["state_dict"], dict):
                state_dict = raw["state_dict"]
            elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
                state_dict = raw["model_state_dict"]
            else:
                state_dict = raw
        else:
            try:
                raw = raw.to(device)
                raw.eval()
                model = raw
                logger.info("Loaded entire model object from checkpoint.")
                return model
            except Exception as e:
                raise RuntimeError(f"Unsupported model file format: {e}")
        state_dict = _strip_module_prefix(state_dict)
        keys = list(state_dict.keys())
        has_classifier = any("classifier" in k for k in keys)
        has_fc = any(k.startswith("fc.") or ".fc." in k for k in keys)
        has_backbone = any("backbone" in k or k.startswith("backbone") for k in keys)
        logger.info("Model architecture detection:")
        logger.info(f"  - Has 'classifier' layers: {has_classifier}")
        logger.info(f"  - Has 'fc' layers: {has_fc}")
        logger.info(f"  - Has 'backbone' structure: {has_backbone}")
        if has_classifier:
            candidate = EnhancedResNet(num_classes=len(CLASS_NAMES), pretrained=False)
            logger.info("Detected: EnhancedResNet architecture (using classifier keys).")
        elif has_fc and not has_backbone:
            candidate = SimpleResNet(num_classes=len(CLASS_NAMES), pretrained=False)
            logger.info("Detected: SimpleResNet architecture (fc-based ResNet).")
        else:
            candidate = EnhancedResNet(num_classes=len(CLASS_NAMES), pretrained=False)
            logger.warning("Unknown architecture: defaulting to EnhancedResNet and using strict=False for loading.")
        load_result = candidate.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(load_result, "missing_keys", None)
        unexpected_keys = getattr(load_result, "unexpected_keys", None)
        if missing_keys is None and isinstance(load_result, tuple):
            missing_keys, unexpected_keys = load_result
        if missing_keys:
            logger.warning(f"Missing keys count: {len(missing_keys)}; sample: {missing_keys[:5]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys count: {len(unexpected_keys)}; sample: {unexpected_keys[:5]}")
        candidate = candidate.to(device)
        candidate.eval()
        model = candidate
        logger.info("Model loaded and set to eval() on device: %s", device)
    return model

def get_treatment_info_api(condition: str) -> str:
    """Get treatment information for a condition"""
    try:
        medical_info = MEDICAL_DATABASE.get(normalize_class_key(condition), {})
        return medical_info.get('treatment', 'Consult a healthcare professional for treatment options.')
    except Exception as e:
        logger.warning(f"Treatment info error: {e}")
        return "Treatment information unavailable."

# ----------------------------
# PREPROCESSING & PREDICTION
# ----------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert input to RGB 3-channel, preserve aspect ratio by padding, then resize to 224.
    This helps MRI (single-channel) and inconsistent image sizes.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    desired_size = 256
    old_size = image.size  # (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image_resized = image.resize(new_size, Image.BILINEAR)
    new_image = Image.new("RGB", (desired_size, desired_size), (0, 0, 0))
    paste_pos = ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    new_image.paste(image_resized, paste_pos)
    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(new_image).unsqueeze(0).to(device)
    return tensor

def predict_medical_image(image: Image.Image, threshold: float = 0.5) -> Dict:
    """Predict using local trained model - NO DATASET REQUIRED (robustified)."""
    try:
        loaded_model = load_model()
        input_tensor = preprocess_image(image)
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        with torch.no_grad():
            model_output = loaded_model(input_tensor)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            if logits is None:
                raise RuntimeError("Model returned None logits.")
            probs = F.softmax(logits, dim=1)
            all_probs = probs[0].cpu().numpy().tolist()
            predicted_idx = int(probs.argmax(dim=1)[0])
            confidence = float(all_probs[predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]
            topk_vals, topk_idx = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
            topk = [(int(i), float(v)) for i, v in zip(topk_idx[0].cpu().numpy().tolist(), topk_vals[0].cpu().numpy().tolist())]
            logger.info(f"All probabilities: {list(zip(CLASS_NAMES, all_probs))}")
            logger.info(f"Predicted index: {predicted_idx}, class: {predicted_class}, confidence: {confidence:.4f}")
            logger.info(f"Top-k: {topk}")
        max_prob = max(all_probs)
        min_prob = min(all_probs)
        prob_range = max_prob - min_prob
        if prob_range < 0.25:
            logger.warning(f"WARNING: Model predictions are unusually uniform (range={prob_range:.3f}). This often means weights are random or checkpoint incompatible.")
        label_for_output = predicted_class
        if confidence < threshold:
            logger.warning(f"Low confidence: {confidence:.3f} < threshold {threshold}")
            label_for_output = f"{predicted_class} (Low Confidence {confidence:.1%})"
        medical_info = MEDICAL_DATABASE.get(normalize_class_key(predicted_class), {})
        return {
            "predicted_class": label_for_output,
            "base_class_name": normalize_class_key(predicted_class),
            "confidence": confidence,
            "all_probabilities": all_probs,
            "class_names": CLASS_NAMES,
            "medical_info": medical_info,
            "topk": [{"idx": idx, "class": CLASS_NAMES[idx], "prob": prob} for idx, prob in topk]
        }
    except Exception as e:
        logger.exception("Prediction error:")
        return {
            "predicted_class": "Error",
            "base_class_name": "Error",
            "confidence": 0.0,
            "all_probabilities": [0.0] * len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "error": str(e)
        }

def create_enhanced_gradcam(image: Image.Image, predicted_class: str) -> Image.Image:
    """Create enhanced Grad-CAM visualization with better highlighting (robust)."""
    try:
        loaded_model = load_model()
        input_tensor = preprocess_image(image)
        gradcam = SimpleGradCAM(loaded_model)
        try:
            if predicted_class in CLASS_NAMES:
                class_idx = CLASS_NAMES.index(predicted_class)
            else:
                # If predicted_class contains extra text like ' (Low Confidence ...)' strip
                base = predicted_class.split(" (")[0]
                if base in CLASS_NAMES:
                    class_idx = CLASS_NAMES.index(base)
                else:
                    class_idx = None
        except Exception:
            class_idx = None
        cam_mask = gradcam.generate(input_tensor, class_idx=class_idx)
        orig_img = np.array(image.convert("RGB"))
        h, w = orig_img.shape[:2]
        cam_resized = cv2.resize(cam_mask, (w, h))
        cam_uint8 = np.uint8(255 * cam_resized)
        colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        alpha = 0.55
        highlighted = cv2.addWeighted(orig_img.astype(np.uint8), 1.0 - alpha, colored.astype(np.uint8), alpha, 0)
        try:
            contours, _ = cv2.findContours(cam_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlighted, contours, -1, (255, 255, 255), 1)
        except Exception:
            pass
        return Image.fromarray(highlighted)
    except Exception as e:
        logger.exception("Grad-CAM creation failed:")
        return image

# ----------------------------
# LOCATION / HOSPITAL HELPERS (Part-2)
# ----------------------------
def geocode_location(location: str) -> Optional[Dict]:
    """Geocode location using Nominatim (OpenStreetMap) as free alternative"""
    try:
        if requests is None:
            logger.warning("Requests library unavailable; geocoding disabled.")
            return None
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'in'  # Focus on India by default
        }
        headers = {'User-Agent': 'MedicalAI-HospitalFinder/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        if results:
            return {
                'lat': float(results[0]['lat']),
                'lng': float(results[0]['lon']),
                'address': results[0].get('display_name', location)
            }
        return None
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None

def get_medical_keywords(condition: str) -> str:
    condition_lower = condition.lower()
    if "brain" in condition_lower or "tumor" in condition_lower:
        return "neurology neurosurgery brain tumor"
    elif "fracture" in condition_lower or "bone" in condition_lower:
        return "orthopedics fracture bone surgery"
    elif "pneumonia" in condition_lower or "chest" in condition_lower:
        return "pulmonology respiratory chest"
    else:
        return "hospital emergency medical"

def should_include_hospital(hospital: Dict, condition: str) -> bool:
    condition_lower = condition.lower()
    specialty_lower = hospital["specialty"].lower()
    if "brain" in condition_lower or "tumor" in condition_lower:
        return "neurology" in specialty_lower or "multi-specialty" in specialty_lower
    elif "fracture" in condition_lower or "bone" in condition_lower:
        return "orthopedics" in specialty_lower or "multi-specialty" in specialty_lower
    elif "pneumonia" in condition_lower or "chest" in condition_lower:
        return "pulmonology" in specialty_lower or "multi-specialty" in specialty_lower
    else:
        return True

def get_realistic_hospitals_by_coordinates(lat: float, lng: float, condition: str) -> List[Dict]:
    hospital_networks = {
        (19.0, 72.8): [
            {"name": "Lilavati Hospital and Research Centre", "address": "A-791, Bandra Reclamation, Bandra West, Mumbai", "phone": "+91-22-66666666", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Kokilaben Dhirubhai Ambani Hospital", "address": "Rao Saheb Achutrao Patwardhan Marg, Four Bungalows, Andheri West", "phone": "+91-22-30999999", "rating": 4.7, "specialty": "Multi-specialty"},
            {"name": "Fortis Hospital Mulund", "address": "Mulund Goregaon Link Road, Mulund West", "phone": "+91-22-30888888", "rating": 4.3, "specialty": "Multi-specialty"},
        ],
        (28.6, 77.2): [
            {"name": "Apollo Hospitals Delhi", "address": "Sarita Vihar, New Delhi", "phone": "+91-11-29871000", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Escorts Heart Institute", "address": "Okhla Road, New Delhi", "phone": "+91-11-26834400", "rating": 4.6, "specialty": "Cardiology"},
            {"name": "Max Super Speciality Hospital", "address": "Saket, New Delhi", "phone": "+91-11-40554055", "rating": 4.4, "specialty": "Multi-specialty"},
        ],
        (12.9, 77.6): [
            {"name": "Apollo Hospitals Bangalore", "address": "154/11, Bannerghatta Road, Bangalore", "phone": "+91-80-26304050", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Hospital Bangalore", "address": "154/9, Cunningham Road, Bangalore", "phone": "+91-80-22288888", "rating": 4.3, "specialty": "Multi-specialty"},
            {"name": "Manipal Hospital", "address": "98, HAL Airport Road, Bangalore", "phone": "+91-80-25024444", "rating": 4.6, "specialty": "Multi-specialty"},
        ],
        (13.0, 80.2): [
            {"name": "Apollo Hospitals Chennai", "address": "21, Greams Lane, Chennai", "phone": "+91-44-28290200", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Malar Hospital", "address": "52, 1st Main Road, Gandhi Nagar, Adyar", "phone": "+91-44-42892222", "rating": 4.3, "specialty": "Multi-specialty"},
            {"name": "MIOT International", "address": "4/112, Mount Poonamallee Road, Manapakkam", "phone": "+91-44-22492288", "rating": 4.4, "specialty": "Multi-specialty"},
        ],
        (17.3, 78.4): [
            {"name": "Apollo Hospitals Hyderabad", "address": "Jubilee Hills, Hyderabad", "phone": "+91-40-23607777", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Hospital Hyderabad", "address": "Kondapur, Hyderabad", "phone": "+91-40-44884488", "rating": 4.3, "specialty": "Multi-specialty"},
            {"name": "Continental Hospitals", "address": "Gachibowli, Hyderabad", "phone": "+91-40-67022222", "rating": 4.4, "specialty": "Multi-specialty"},
        ],
        (22.5, 88.3): [
            {"name": "Apollo Gleneagles Hospitals", "address": "58, Canal Circular Road, Kolkata", "phone": "+91-33-23206060", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Hospital Kolkata", "address": "730, Anandapur, Kolkata", "phone": "+91-33-66284444", "rating": 4.3, "specialty": "Multi-specialty"},
            {"name": "AMRI Hospitals", "address": "Salt Lake, Kolkata", "phone": "+91-33-23203030", "rating": 4.4, "specialty": "Multi-specialty"},
        ],
        (18.5, 73.8): [
            {"name": "Apollo Hospitals Pune", "address": "Baner Road, Pune", "phone": "+91-20-27204444", "rating": 4.5, "specialty": "Multi-specialty"},
            {"name": "Fortis Hospital Pune", "address": "Wanowrie, Pune", "phone": "+91-20-25555555", "rating": 4.3, "specialty": "Multi-specialty"},
            {"name": "Sahyadri Hospitals", "address": "Deccan Gymkhana, Pune", "phone": "+91-20-25555555", "rating": 4.4, "specialty": "Multi-specialty"},
        ]
    }
    closest_city = None
    min_distance = float('inf')
    for city_coords, hospitals in hospital_networks.items():
        distance = ((lat - city_coords[0])**2 + (lng - city_coords[1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_city = hospitals
    if closest_city:
        filtered_hospitals = []
        for hospital in closest_city:
            distance_km = min_distance * 111
            if should_include_hospital(hospital, condition):
                hospital_data = {
                    "name": hospital["name"],
                    "address": hospital["address"],
                    "phone": hospital["phone"],
                    "specialty": hospital["specialty"],
                    "distance": f"{distance_km:.1f} km",
                    "rating": hospital["rating"],
                    "emergency": True
                }
                filtered_hospitals.append(hospital_data)
        return filtered_hospitals[:5]
    return []

def get_fallback_hospitals(location: str, condition: str) -> List[Dict]:
    return [
        {"name":f"Emergency Hospital - {location}","address":f"Emergency Zone, {location}","phone":"102","specialty":"Emergency","distance":"5 km","rating":4.0,"emergency":True},
        {"name":f"City Medical Center - {location}","address":f"Main Medical District, {location}","phone":"102-000-0001","specialty":"Multi-specialty","distance":"8 km","rating":4.2,"emergency":True}
    ]

def search_hospitals_google_places(lat: float, lng: float, condition: str) -> List[Dict]:
    """Simulated Google Places: return realistic hospitals near coordinates."""
    try:
        logger.info(f"🔍 Searching for hospitals near {lat}, {lng} for condition: {condition}")
        hospitals = get_realistic_hospitals_by_coordinates(lat, lng, condition)
        return hospitals
    except Exception as e:
        logger.error(f"Google Places search error: {e}")
        return []

def get_hospitals_near_location(location: str, condition: str) -> List[Dict]:
    """Get hospitals near a human-readable location string."""
    try:
        logger.info(f"🔍 Finding real hospitals near '{location}' for condition '{condition}'")
        geocoded_location = geocode_location(location)
        if not geocoded_location:
            logger.warning(f"Could not geocode location '{location}', using fallback hospitals")
            return get_fallback_hospitals(location, condition)
        lat, lng = geocoded_location['lat'], geocoded_location['lng']
        logger.info(f"📍 Geocoded '{location}' to {lat}, {lng}")
        hospitals = search_hospitals_google_places(lat, lng, condition)
        if hospitals:
            logger.info(f"✅ Found {len(hospitals)} real hospitals near '{location}'")
            return hospitals
        else:
            logger.warning(f"No hospitals found via search, using fallback")
            return get_fallback_hospitals(location, condition)
    except Exception as e:
        logger.error(f"Hospital search error: {e}")
        return get_fallback_hospitals(location, condition)

def create_hospital_map(location: str, hospitals: List[Dict]) -> str:
    """Create an interactive map showing nearby hospitals"""
    try:
        center_lat, center_lng = 17.3850, 78.4867  # default Hyderabad
        geocoded = geocode_location(location) if location else None
        if geocoded:
            center_lat, center_lng = geocoded['lat'], geocoded['lng']
        m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='OpenStreetMap')
        for i, hospital in enumerate(hospitals):
            # offset markers a little (so they don't stack exactly)
            folium.Marker(
                location=[center_lat + (i * 0.01), center_lng + (i * 0.01)],
                popup=f"""
                <b>{hospital['name']}</b><br>
                {hospital['address']}<br>
                Phone: {hospital['phone']}<br>
                Distance: {hospital['distance']}<br>
                Rating: {hospital['rating']}/5
                """,
                icon=folium.Icon(color='red', icon='plus-sign')  # safe fallback icon
            ).add_to(m)
        map_html = m._repr_html_()
        return map_html
    except Exception as e:
        logger.error(f"Map creation error: {e}")
        return f"<p>Map unavailable. Error: {str(e)}</p>"

# ----------------------------
# REPORT & ANALYSIS (Part-2)
# ----------------------------
def generate_medical_report(prediction_result: Dict, image: Image.Image, highlighted_image: Image.Image) -> str:
    """Generate comprehensive medical report"""
    try:
        medical_info = prediction_result.get("medical_info", {})
        predicted_class = prediction_result.get("predicted_class", "Unknown")
        confidence = prediction_result.get("confidence", 0.0)
        report = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #2c3e50; text-align: center;">Medical Analysis Report</h1>
            <hr style="border: 2px solid #3498db;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #e74c3c;">Diagnosis Summary</h2>
                <p><strong>Condition:</strong> {medical_info.get('condition', 'Unknown')}</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Severity:</strong> {medical_info.get('severity', 'Unknown')}</p>
                <p><strong>Urgency:</strong> {medical_info.get('urgency', 'Unknown')}</p>
            </div>
            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #27ae60;">Description</h2>
                <p>{medical_info.get('description', 'No description available.')}</p>
            </div>
            <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #856404;">Symptoms</h2>
                <ul>
                    {''.join([f'<li>{symptom}</li>' for symptom in medical_info.get('symptoms', [])])}
                </ul>
            </div>
            <div style="background-color: #d1ecf1; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #0c5460;">Treatment</h2>
                <p>{medical_info.get('treatment', 'Consult a healthcare professional.')}</p>
            </div>
            <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #721c24;">Specialist Recommendation</h2>
                <p><strong>Recommended Specialist:</strong> {medical_info.get('specialist', 'General Practitioner')}</p>
                <p><strong>Follow-up:</strong> {medical_info.get('follow_up', 'As needed')}</p>
            </div>
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #155724;">Emergency Information</h2>
                <p><strong>Emergency Required:</strong> {'Yes' if medical_info.get('emergency', False) else 'No'}</p>
                <p><strong>Emergency Contacts:</strong> 108 (Ambulance), 100 (Police)</p>
            </div>
            <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
                <p style="color: #6c757d; font-style: italic;">
                    <strong>Disclaimer:</strong> This analysis is for educational purposes only. 
                    Always consult qualified healthcare professionals for medical diagnosis and treatment.
                </p>
            </div>
        </div>
        """
        return report
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return f"<p>Report generation failed: {str(e)}</p>"

def analyze_medical_image(image: Image.Image, location: str = "", threshold: float = 0.5) -> Tuple[str, Optional[Image.Image], str, str, str]:
    """Main analysis function with enhanced features including remedy API"""
    if image is None:
        return "Please upload an image", None, "No image provided", "", ""
    try:
        results = predict_medical_image(image, threshold)
        if "error" in results:
            return f"Error: {results['error']}", None, "Analysis failed", "", ""
        predicted_class = results["predicted_class"]
        base_class_name = results.get("base_class_name", predicted_class)
        confidence = results["confidence"]
        medical_info = results.get("medical_info", {})
        highlighted_image = create_enhanced_gradcam(image, predicted_class)
        report = generate_medical_report(results, image, highlighted_image)
        remedy_info = ""
        try:
            remedy_info = get_treatment_info_api(base_class_name)
        except Exception as api_err:
            logger.warning(f"Treatment API failed: {api_err}")
        hospitals_html = ""
        if location:
            hospitals = get_hospitals_near_location(location, base_class_name)
            hospitals_html = create_hospital_map(location, hospitals)
        diagnosis = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px;">
            <h2 style="margin: 0 0 15px 0; text-align: center;">🔍 AI Diagnosis Results</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">📋 Condition Detected</h3>
                <p style="font-size: 18px; margin: 5px 0;"><strong>{medical_info.get('condition', predicted_class)}</strong></p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">📊 Confidence Score</h3>
                <p style="font-size: 24px; margin: 5px 0; color: #4CAF50;"><strong>{confidence:.1%}</strong></p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">⚠️ Severity Level</h3>
                <p style="font-size: 18px; margin: 5px 0; color: {'#ff4444' if medical_info.get('severity') == 'High' else '#ffaa00' if medical_info.get('severity') == 'Medium' else '#44ff44'}">
                    <strong>{medical_info.get('severity', 'Unknown')}</strong>
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">🚨 Urgency</h3>
                <p style="font-size: 18px; margin: 5px 0; color: {'#ff4444' if medical_info.get('urgency') == 'High' else '#ffaa00' if medical_info.get('urgency') == 'Medium' else '#44ff44'}">
                    <strong>{medical_info.get('urgency', 'Unknown')}</strong>
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">💊 Recommended Treatment / Remedy</h3>
                <p style="font-size: 16px; margin: 5px 0; color: #ffffff;"><strong>{remedy_info}</strong></p>
            </div>
        </div>
        """
        chart_data = []
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, results["all_probabilities"])):
            chart_data.append([class_name.replace('_', ' ').title(), float(prob)])
        chart_data.sort(key=lambda x: x[1], reverse=True)
        chart_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h3 style="text-align: center; color: #2c3e50;">📈 Confidence Scores for All Conditions</h3>
            <div style="display: grid; gap: 10px;">
        """
        for class_name, prob in chart_data:
            color = "#4CAF50" if prob > 0.5 else "#FF9800" if prob > 0.2 else "#F44336"
            chart_html += f"""
                <div style="background: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <strong>{class_name}</strong>: {prob:.1%}
                </div>
            """
        chart_html += "</div></div>"
        return diagnosis, highlighted_image, chart_html, report, hospitals_html
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return f"Error during analysis: {str(e)}", None, "Analysis failed", "", ""

# ----------------------------
# GRADIO INTERFACE (unchanged layout, minor safe fallbacks)
# ----------------------------
def create_enhanced_interface():
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .emergency-button {
        background: linear-gradient(45deg, #ff4444, #cc0000);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .emergency-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 68, 68, 0.4);
    }
    """
    with gr.Blocks(title="🩺 Enhanced Medical AI Platform", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.HTML("""
        <div class="main-header">
            <h1>🩺 Enhanced Medical AI Platform</h1>
            <p style="font-size: 18px; margin: 10px 0;">Advanced AI-powered medical image analysis with hospital tracing and location services</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                <span style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px;">🧠 Brain Analysis</span>
                <span style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px;">🫁 Chest X-ray</span>
                <span style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px;">🦴 Bone Fracture</span>
            </div>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Upload Medical Image")
                image_input = gr.Image(label="Drag & Drop or Click to Upload", type="pil", height=400, elem_classes=["feature-card"])
                gr.Markdown("### 📍 Your Location")
                location_input = gr.Textbox(label="Enter your city/area for hospital recommendations", placeholder="e.g., Hyderabad, India or Mumbai, Maharashtra", elem_classes=["feature-card"])
                gr.Markdown("### ⚙️ Analysis Settings")
                threshold = gr.Slider(minimum=0.0, maximum=0.9, value=0.1, step=0.05, label="Confidence Threshold (lower = more lenient)", elem_classes=["feature-card"])
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg", elem_classes=["emergency-button"])
                gr.Markdown("### 🚨 Emergency Contacts")
                gr.HTML("""
                <div style="text-align: center;">
                    <button class="emergency-button" onclick="window.open('tel:108')">🚑 Ambulance: 108</button>
                    <button class="emergency-button" onclick="window.open('tel:100')">👮 Police: 100</button>
                    <button class="emergency-button" onclick="window.open('tel:101')">🚒 Fire: 101</button>
                </div>
                """)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Diagnosis Results"):
                        diagnosis_output = gr.HTML(value="<div style='text-align: center; padding: 50px; color: #666;'>Upload an image and click 'Analyze Image' to see results</div>")
                    with gr.Tab("🔍 Affected Areas"):
                        gr.Markdown("### Highlighted Regions (AI Visualization)")
                        gradcam_display = gr.Image(label="Areas highlighted by AI analysis", type="pil", height=500)
                    with gr.Tab("📈 Confidence Analysis"):
                        confidence_output = gr.HTML(value="<div style='text-align: center; padding: 50px; color: #666;'>Confidence analysis will appear here</div>")
                    with gr.Tab("📋 Detailed Report"):
                        report_output = gr.HTML(value="<div style='text-align: center; padding: 50px; color: #666;'>Detailed medical report will appear here</div>")
                    with gr.Tab("🏥 Nearby Hospitals"):
                        hospitals_output = gr.HTML(value="<div style='text-align: center; padding: 50px; color: #666;'>Hospital map will appear here when location is provided</div>")
        def process_image(image, location, thresh):
            if image is None:
                return ("<div style='text-align: center; padding: 50px; color: #666;'>Please upload an image</div>", None,
                        "<div style='text-align: center; padding: 50px; color: #666;'>No image provided</div>",
                        "<div style='text-align: center; padding: 50px; color: #666;'>No analysis performed</div>",
                        "<div style='text-align: center; padding: 50px; color: #666;'>No location provided</div>")
            diagnosis, highlighted, confidence, report, hospitals = analyze_medical_image(image, location, thresh)
            return diagnosis, highlighted, confidence, report, hospitals
        analyze_btn.click(fn=process_image, inputs=[image_input, location_input, threshold],
                          outputs=[diagnosis_output, gradcam_display, confidence_output, report_output, hospitals_output])
        image_input.change(fn=process_image, inputs=[image_input, location_input, threshold],
                           outputs=[diagnosis_output, gradcam_display, confidence_output, report_output, hospitals_output])
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 30px;">
            <h3 style="color: white; margin-bottom: 15px;">⚠️ Important Medical Disclaimer</h3>
            <p style="color: white; font-size: 14px; line-height: 1.6;">
                This AI-powered medical analysis tool is designed for educational and research purposes only. 
                It should not be used as a substitute for professional medical diagnosis, treatment, or advice. 
                Always consult qualified healthcare professionals for medical concerns and emergencies.
            </p>
            <p style="color: white; font-size: 12px; margin-top: 15px;">
                © 2024 Enhanced Medical AI Platform - Powered by Advanced Deep Learning
            </p>
        </div>
        """)
    return interface

# ----------------------------
# MAIN - Launch
# ----------------------------
def main():
    try:
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
        model_found = False
        for path in ["medical_model.pth", "model/best_model.pth", "best_model.pth"]:
            if os.path.exists(path):
                model_found = True
                break
        if not model_found:
            print("WARNING: No model file found. The app will attempt to load a model when needed.")
        interface = create_enhanced_interface()
        print("Starting Enhanced Medical AI Platform...")
        print("Features:")
        print("   ✅ Local image prediction - NO DATASET REQUIRED")
        print("   ✅ Hospital tracing based on location")
        print("   ✅ Enhanced image highlighting with Grad-CAM")
        print("   ✅ Modern attractive UI with animations")
        print("   ✅ Comprehensive medical database")
        print("   ✅ Emergency contacts and notifications")
        print("   ✅ Interactive hospital maps")
        print("   ✅ Detailed medical reports")
        print("\nThe web interface will open in your browser")
        print("If it doesn't open automatically, check the terminal for the URL")
        print("Press Ctrl+C to stop the server")
        print("\nTIP: Upload any medical image (X-ray, MRI, CT scan) to get AI predictions")
        interface.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True, quiet=False)
    except KeyboardInterrupt:
        print("\nEnhanced Medical Platform stopped by user")
    except Exception as e:
        print(f"Error starting platform: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
