#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical AI App - Regenerated, corrected, UI-polished, with Grad-CAM++ (Option 2)
Loads model from: medical_model.pth (change MODEL_PATH below if needed)
Run: python medical_final_app_regenerated.py
"""

import os
import sys
import logging
import traceback
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
import requests
import math
import json

# -------------------- Config --------------------
MODEL_PATH = "medical_model.pth"   # Should match CHECKPOINT_PATH in train.py
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalAI")

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------- Fallback class names (will be overridden if checkpoint contains mapping) ----------
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

def normalize_class_key(key: str) -> str:
    if key is None:
        return key
    return key.replace("/", "_").replace(" ", "_")

def infer_modality_from_class(class_name: str) -> str:
    """Infer modality from class name (works with both dataset names and display names)"""
    class_lower = class_name.lower()
    # Check for display name format (with prefixes)
    if class_name.startswith("brain_tumor/") or ("brain" in class_lower and "tumor" in class_lower):
        return "brain"
    if class_name.startswith("bone_fracture/") or "fracture" in class_lower or ("bone" in class_lower and "fracture" in class_lower):
        return "bone"
    if class_name.startswith("chest_xray/") or ("chest" in class_lower and "xray" in class_lower):
        return "chest"
    # Check for dataset name format (simple names)
    if class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        return "brain"
    if class_name in ["fractured", "not fractured"]:
        return "bone"
    if class_name in ["NORMAL", "PNEUMONIA"]:
        return "chest"
    return "generic"

# -------------------- Medical DB (kept from your original, normalized keys) --------------------
MEDICAL_DATABASE = {
    "bone_fracture_fractured": {
        "condition": "Bone Fracture Detected",
        "severity":"High",
        "urgency":"Immediate",
        "description":"A break or crack in a bone has been detected in the X-ray image.",
        "symptoms":["Pain","Swelling","Bruising","Deformity","Loss of function"],
        "treatment":"Immobilization with cast or splint, pain management, and possible surgery for complex fractures.",
        "specialist":"Orthopedic Surgeon",
        "emergency":True,
        "follow_up":"Immediate medical attention required"
    },
    "bone_fracture_not_fractured": {
        "condition":"No Bone Fracture",
        "severity":"Low",
        "urgency":"Low",
        "description":"No signs of bone fracture detected. Bone structure appears normal.",
        "symptoms":["May have soft tissue injury"],
        "treatment":"Rest, ice, compression, elevation (RICE) for soft tissue injuries.",
        "specialist":"General Practitioner",
        "emergency":False,
        "follow_up":"Monitor for persistent pain"
    },
    "brain_tumor_glioma": {
        "condition":"Glioma Brain Tumor",
        "severity":"High",
        "urgency":"High",
        "description":"A glioma (malignant brain tumor) has been detected in the MRI scan.",
        "symptoms":["Headaches","Seizures","Cognitive changes","Motor weakness","Vision problems"],
        "treatment":"Surgery, radiation therapy, chemotherapy, and targeted therapy.",
        "specialist":"Neurosurgeon, Oncologist",
        "emergency":True,
        "follow_up":"Immediate neurological consultation required"
    },
    "brain_tumor_meningioma": {
        "condition":"Meningioma Brain Tumor",
        "severity":"Medium",
        "urgency":"Medium",
        "description":"A meningioma (usually benign brain tumor) has been detected.",
        "symptoms":["Headaches","Seizures","Vision changes","Memory problems"],
        "treatment":"Surgical removal, radiation therapy, or monitoring for small tumors.",
        "specialist":"Neurosurgeon",
        "emergency":False,
        "follow_up":"Neurosurgical consultation within 1-2 weeks"
    },
    "brain_tumor_notumor": {
        "condition":"No Brain Tumor",
        "severity":"Low",
        "urgency":"Low",
        "description":"No brain tumor detected. Brain structure appears normal.",
        "symptoms":["None"],
        "treatment":"No treatment needed. Continue regular health monitoring.",
        "specialist":"General Practitioner",
        "emergency":False,
        "follow_up":"Routine follow-up as needed"
    },
    "brain_tumor_pituitary": {
        "condition":"Pituitary Tumor",
        "severity":"Medium",
        "urgency":"Medium",
        "description":"A pituitary tumor has been detected. May affect hormone production.",
        "symptoms":["Hormonal imbalances","Vision problems","Headaches","Fatigue"],
        "treatment":"Surgical removal, medication, or radiation therapy.",
        "specialist":"Endocrinologist, Neurosurgeon",
        "emergency":False,
        "follow_up":"Endocrinological consultation within 1 week"
    },
    "chest_xray_NORMAL": {
        "condition":"Normal Chest X-ray",
        "severity":"Low",
        "urgency":"Low",
        "description":"No signs of pneumonia or other lung abnormalities detected.",
        "symptoms":["None"],
        "treatment":"No treatment needed. Maintain good respiratory health.",
        "specialist":"General Practitioner",
        "emergency":False,
        "follow_up":"Routine health checkup"
    },
    "chest_xray_PNEUMONIA": {
        "condition":"Pneumonia Detected",
        "severity":"High",
        "urgency":"High",
        "description":"Signs of pneumonia (lung infection) detected in the chest X-ray.",
        "symptoms":["Cough","Fever","Difficulty breathing","Chest pain","Fatigue"],
        "treatment":"Antibiotics, rest, fluids, and possible hospitalization for severe cases.",
        "specialist":"Pulmonologist, Infectious Disease Specialist",
        "emergency":True,
        "follow_up":"Immediate medical attention required"
    }
}

# -------------------- Model classes (matching train.py) --------------------
class EfficientNetModel(nn.Module):
    """EfficientNet model matching train.py - returns single tensor like train.py"""
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # Replace classifier (matching train.py)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        # Return single tensor like train.py (not tuple)
        out = self.model(x)
        return out

class EnhancedResNet(nn.Module):
    """ResNet model (fallback)"""
    def __init__(self, num_classes, pretrained=True):
        super(EnhancedResNet, self).__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        num_features = self.base.fc.in_features
        
        # Replace FC layer with custom classifier
        self.base.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        out = self.base(x)
        # Return tuple (logits, feature_map) for compatibility with Grad-CAM
        batch = x.size(0)
        feature = torch.zeros(batch, 512, 7, 7, device=x.device, dtype=x.dtype)
        return out, feature

class SimpleResNet(nn.Module):
    """Simple ResNet (fallback)"""
    def __init__(self, num_classes, pretrained=True):
        super(SimpleResNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        numf = self.model.fc.in_features
        self.model.fc = nn.Linear(numf, num_classes)
    def forward(self, x):
        out = self.model(x)
        batch = x.size(0)
        feature = torch.zeros(batch, 512, 7, 7, device=x.device, dtype=x.dtype)
        return out, feature

# -------------------- Grad-CAM++ implementation (enhanced) --------------------
class GradCAMPlusPlus:
    """
    Implementation of Grad-CAM++:
    - forward hook captures activations
    - register hook on activation to capture gradients
    - compute alpha weights as in Grad-CAM++
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        # forward hook
        def forward_hook(module, inp, out):
            self.activations = out.detach()
            # attach grad hook on activation tensor
            def save_grad(grad):
                self.gradients = grad.detach()
            try:
                out.register_hook(save_grad)
            except Exception:
                pass
        h = target_layer.register_forward_hook(forward_hook)
        self.handles.append(h)
    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: [1,C,H,W]
        returns: cam resized to (H,W) normalized 0..1
        """
        self.model.zero_grad()
        # Handle both tuple (ResNet) and single tensor (EfficientNet) outputs
        model_out = self.model(input_tensor)
        if isinstance(model_out, tuple):
            out = model_out[0]
        else:
            out = model_out
        if class_idx is None:
            class_idx = int(out.argmax(dim=1)[0])
        score = out[0, class_idx]
        score.backward(retain_graph=True)
        # grads: [B, C, H', W'] or [C,H',W']
        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            # fallback: zeros
            h, w = input_tensor.size(2), input_tensor.size(3)
            return np.zeros((h,w), dtype=np.float32)
        if grads.dim() == 4:
            grads = grads[0]
        if acts.dim() == 4:
            acts = acts[0]
        # compute Grad-CAM++ weights
        # per https://arxiv.org/abs/1710.11063 (approx)
        eps = 1e-8
        grads_pow_2 = grads ** 2
        grads_pow_3 = grads ** 3
        # global sum over spatial
        sum_grads = torch.sum(grads, dim=(1,2), keepdim=True)  # [C,1,1]
        # alpha numerator and denominator
        numerator = grads_pow_2
        denominator = 2 * grads_pow_2 + torch.sum(acts * grads_pow_3, dim=(1,2), keepdim=True)
        denominator = denominator + eps
        alphas = numerator / denominator  # [C,H',W']
        # weights = relu(sum over spatial of alphas * relu(grads))
        relu_grads = torch.relu(grads)
        weights = torch.sum(alphas * relu_grads, dim=(1,2))  # [C]
        # weighted combination
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)  # [H',W']
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= (cam.max() + eps)
        cam_np = cam.cpu().numpy()
        # resize to input spatial
        out_h, out_w = input_tensor.size(2), input_tensor.size(3)
        cam_resized = cv2.resize(cam_np, (out_w, out_h))
        # release hooks
        self.remove_hooks()
        return cam_resized

# -------------------- Model loading & checkpoint parsing --------------------
_model_cache = None
_class_names = CLASS_NAMES.copy()  # Will be replaced by checkpoint
_display_names = None  # Display names for UI (with prefixes)
_name_mapping = None  # Mapping from dataset names to display names
_class_to_idx = None
_checkpoint_norm = None

def _load_checkpoint(path: str):
    logger.info(f"Loading checkpoint: {path}")
    cp = torch.load(path, map_location=device)
    # try to extract useful metadata
    state_dict = None
    if isinstance(cp, dict):
        # support many checkpoint formats
        if "state_dict" in cp:
            state_dict = cp["state_dict"]
        elif "model_state_dict" in cp:
            state_dict = cp["model_state_dict"]
        else:
            # maybe cp itself is state_dict (keys-> tensors)
            # check for tensor values heuristic
            if all(isinstance(v, torch.Tensor) for v in cp.values()):
                state_dict = cp
            else:
                # Extract metadata from checkpoint
                if "class_names" in cp and isinstance(cp["class_names"], (list,tuple)):
                    global _class_names
                    _class_names = list(cp["class_names"])
                    logger.info(f"Loaded class_names (dataset names) from checkpoint: {_class_names}")
                if "display_names" in cp and isinstance(cp["display_names"], (list,tuple)):
                    global _display_names
                    _display_names = list(cp["display_names"])
                    logger.info(f"Loaded display_names from checkpoint: {_display_names}")
                if "name_mapping" in cp and isinstance(cp["name_mapping"], dict):
                    global _name_mapping
                    _name_mapping = cp["name_mapping"]
                    logger.info(f"Loaded name_mapping from checkpoint: {_name_mapping}")
                if "class_to_idx" in cp:
                    global _class_to_idx
                    _class_to_idx = cp["class_to_idx"]
                    logger.info("Loaded class_to_idx from checkpoint")
                if "norm" in cp and isinstance(cp["norm"], dict):
                    global _checkpoint_norm
                    _checkpoint_norm = cp["norm"]
                    logger.info(f"Loaded normalization from checkpoint: {_checkpoint_norm}")
                # Extract state_dict
                if "state_dict" in cp:
                    state_dict = cp["state_dict"]
                else:
                    # last resort: search for nested dict with tensors
                    for k,v in cp.items():
                        if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                            state_dict = v
                            break
                    # If still None, check if cp itself has tensor values
                    if state_dict is None and all(isinstance(v, torch.Tensor) for v in cp.values()):
                        state_dict = cp
    else:
        # cp may be state_dict already
        if isinstance(cp, dict) and all(isinstance(v, torch.Tensor) for v in cp.values()):
            state_dict = cp
    if state_dict is None:
        # attempt to use cp as state dict
        state_dict = cp
    return state_dict

def load_model() -> nn.Module:
    global _model_cache, _class_names, _class_to_idx, _checkpoint_norm
    if _model_cache is not None:
        return _model_cache
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Place your model there.")
    state_dict = _load_checkpoint(MODEL_PATH)
    # determine class names length
    num_classes = len(_class_names)
    logger.info(f"Loading model with {num_classes} classes: {_class_names}")
    
    # decide architecture heuristically based on checkpoint keys
    keys = list(state_dict.keys())
    logger.info(f"Checkpoint keys sample (first 10): {keys[:10]}")
    
    # Check for EfficientNet - it has "features" and "classifier" but NOT "model." prefix in state_dict
    has_efficientnet_features = any("features" in k for k in keys)
    has_efficientnet_classifier = any("classifier" in k for k in keys)
    has_model_prefix = any(k.startswith("model.") for k in keys)
    has_base_fc = any("base.fc" in k or k.startswith("base.fc") for k in keys)
    has_model_fc = any("model.fc" in k or (k.startswith("fc.") and "base" not in k) for k in keys)
    has_backbone = any("backbone" in k for k in keys)
    
    # EfficientNet from train.py has keys like "features.0.weight", "classifier.1.weight" (no "model." prefix)
    # But our EfficientNetModel wrapper has self.model, so it expects "model.features.0.weight"
    # Check if this is an EfficientNet checkpoint
    is_efficientnet_checkpoint = has_efficientnet_features and has_efficientnet_classifier
    
    # choose architecture - prioritize EfficientNet (matching train.py)
    if is_efficientnet_checkpoint:
        model = EfficientNetModel(num_classes=num_classes, pretrained=False)
        logger.info("Instantiated EfficientNet (matching train.py)")
        # If checkpoint doesn't have "model." prefix, we need to add it to match our wrapper
        if not has_model_prefix:
            logger.info("Checkpoint keys don't have 'model.' prefix - adding prefix to match EfficientNetModel wrapper")
            adjusted_state_dict = {}
            for k, v in state_dict.items():
                adjusted_state_dict[f"model.{k}"] = v
            state_dict = adjusted_state_dict
            logger.info(f"Adjusted state_dict keys sample: {list(adjusted_state_dict.keys())[:5]}")
    elif has_model_prefix and has_efficientnet_classifier:
        # Checkpoint already has "model." prefix (from our wrapper)
        model = EfficientNetModel(num_classes=num_classes, pretrained=False)
        logger.info("Instantiated EfficientNet (checkpoint has model. prefix)")
    elif has_base_fc:
        model = EnhancedResNet(num_classes=num_classes, pretrained=False)
        logger.info("Instantiated EnhancedResNet")
    elif has_model_fc:
        model = SimpleResNet(num_classes=num_classes, pretrained=False)
        logger.info("Instantiated SimpleResNet")
    elif has_classifier or has_backbone:
        model = EnhancedResNet(num_classes=num_classes, pretrained=False)
        logger.info("Instantiated EnhancedResNet (fallback)")
    else:
        # Default to EfficientNet (matches train.py)
        model = EfficientNetModel(num_classes=num_classes, pretrained=False)
        logger.info("Default to EfficientNet")
    
    # load state dict
    try:
        load_res = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded state_dict with strict=False")
        # report missing/unexpected keys if available
        try:
            missing = getattr(load_res, "missing_keys", []) or load_res.get("missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", []) or load_res.get("unexpected_keys", [])
        except Exception:
            # some PyTorch versions return tuple
            if isinstance(load_res, tuple) and len(load_res) == 2:
                missing, unexpected = load_res
            else:
                missing, unexpected = [], []
        if missing:
            logger.warning(f"Missing keys count: {len(missing)} (sample: {missing[:8]})")
        if unexpected:
            logger.warning(f"Unexpected keys count: {len(unexpected)} (sample: {unexpected[:8]})")
        
        # Verify model is actually loaded (not random)
        # Check if classifier weights are non-zero
        if hasattr(model, "model") and hasattr(model.model, "classifier"):
            classifier_weight = model.model.classifier[1].weight
            weight_std = classifier_weight.std().item()
            logger.info(f"Classifier weight std: {weight_std:.6f} (should be > 0.01 for trained model)")
            if weight_std < 0.01:
                logger.warning("WARNING: Model weights appear to be untrained/random! Model may need retraining.")
    except Exception as e:
        logger.error(f"Model.load_state_dict failed: {e}")
        raise
    model = model.to(device)
    model.eval()
    _model_cache = model
    logger.info("Model ready.")
    return model

# -------------------- Preprocessing --------------------
def preprocess_image(image: Image.Image, modality: Optional[str] = None) -> torch.Tensor:
    """Preprocess image to match training pipeline EXACTLY"""
    image = image.convert("RGB")
    # EXACT match to train.py val_tfms: Resize((224, 224)) - NOT Resize(256)+CenterCrop
    tlist = [transforms.Resize((224, 224)), transforms.ToTensor()]
    # prefer checkpoint norm if present (from train.py checkpoint)
    global _checkpoint_norm
    if _checkpoint_norm and isinstance(_checkpoint_norm, dict) and "mean" in _checkpoint_norm and "std" in _checkpoint_norm:
        mean = _checkpoint_norm["mean"]
        std = _checkpoint_norm["std"]
        logger.info(f"Using checkpoint normalization: mean={mean}, std={std}")
    else:
        # Use ImageNet normalization (matching train.py)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        logger.info(f"Using default ImageNet normalization: mean={mean}, std={std}")
    tlist.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(tlist)
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

# -------------------- Modality estimation (same heuristics) --------------------
def _estimate_image_group(image: Image.Image) -> Tuple[str, float]:
    try:
        arr = np.array(image.convert("RGB"))
        h,w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        edges = cv2.Canny((gray*255).astype(np.uint8), 50, 150)
        edge_density = float(edges.mean())
        mid = w//2
        left = gray[:, :mid]
        right = np.fliplr(gray[:, w-mid:])
        sym_h = min(left.shape[0], right.shape[0])
        sym_w = min(left.shape[1], right.shape[1])
        left_c = left[:sym_h, :sym_w]
        right_c = right[:sym_h, :sym_w]
        symmetry = 1.0 - float(np.mean(np.abs(left_c - right_c)))
        brightness = float(gray.mean())
        score_chest = 0.5*symmetry + 0.2*(1.0-edge_density) + 0.3*brightness
        score_bone = 0.6*edge_density + 0.2*(1.0-symmetry) + 0.2*(1.0-brightness)
        score_brain = 0.4*(1.0-edge_density) + 0.3*symmetry + 0.3*(1.0-abs(brightness-0.5))
        scores = {"chest":score_chest, "bone":score_bone, "brain":score_brain}
        detected = max(scores.items(), key=lambda x:x[1])[0]
        vals = sorted(scores.values(), reverse=True)
        gap = max(vals[0] - vals[1], 0.0)
        conf = float(min(max(gap/(vals[0]+1e-6),0.0),1.0))
        logger.info(f"Auto modality: {detected} (conf {conf:.2f}) scores={scores}")
        return detected, conf
    except Exception as e:
        logger.warning(f"Modality estimation failed: {e}")
        return "generic", 0.0

# -------------------- Prediction pipeline --------------------
def build_index_maps(class_names: List[str]):
    class_to_idx = {cn:i for i,cn in enumerate(class_names)}
    norm_to_idx = {normalize_class_key(cn): i for cn,i in class_to_idx.items()}
    return class_to_idx, norm_to_idx

def predict_medical_image(image: Image.Image, threshold: float = 0.3, modality_hint: Optional[str] = None) -> Dict:
    """
    Returns:
      predicted_class: display str (may contain low confidence marker)
      base_class_name: canonical class name from class list
      confidence: float
      all_probabilities: list
      class_names: list
      medical_info: dict
    """
    try:
        # load model & class names (class list may be overridden by checkpoint)
        model = load_model()
        class_names = _class_names  # Dataset folder names (for model predictions)
        display_names = _display_names if _display_names is not None else class_names  # Display names for UI
        
        # If we have a mapping, use it to convert dataset names to display names
        if _name_mapping and display_names == class_names:
            display_names = [_name_mapping.get(name, name) for name in class_names]
        
        class_to_idx, norm_to_idx = build_index_maps(class_names)
        # modality detection
        if modality_hint and modality_hint != "auto":
            modality = modality_hint
            detect_conf = 1.0
        else:
            modality, detect_conf = _estimate_image_group(image)
        # preprocess
        input_t = preprocess_image(image, modality=modality)
        with torch.no_grad():
            out = model(input_t)
            # Handle both tuple (ResNet) and single tensor (EfficientNet) outputs
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            logger.info(f"Model output shape: {logits.shape}, num_classes: {len(class_names)}")
            
            # Get raw probabilities WITHOUT modality masking first
            probs = F.softmax(logits, dim=1)
            all_probs = probs[0].cpu().numpy().tolist()
            
            # Get top predictions for debugging
            top3_probs, top3_indices = torch.topk(probs[0], k=min(3, len(class_names)))
            logger.info(f"Top 3 predictions (raw):")
            for i, (idx, prob) in enumerate(zip(top3_indices.cpu().tolist(), top3_probs.cpu().tolist())):
                logger.info(f"  {i+1}. {display_names[idx]}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Get prediction WITHOUT modality filtering
            pred_idx = int(torch.argmax(logits, dim=1)[0])
            pred_class_dataset = class_names[pred_idx]
            pred_class_display = display_names[pred_idx]
            confidence = float(all_probs[pred_idx])
            
            # Only apply modality filtering if confidence is low AND modality detection is confident
            # This prevents forcing wrong predictions
            pred_mod = infer_modality_from_class(pred_class_dataset)
            if modality != "auto" and modality_hint and modality_hint != "auto":
                # User specified modality - check if prediction matches
                if pred_mod != modality and confidence < 0.7:
                    logger.info(f"Low confidence ({confidence:.2f}) and modality mismatch. Checking modality-specific predictions.")
                    modality_indices = [i for i,cn in enumerate(class_names) if infer_modality_from_class(cn) == modality]
                    if modality_indices:
                        modality_probs = [(i, all_probs[i]) for i in modality_indices]
                        modality_probs.sort(key=lambda x:x[1], reverse=True)
                        if modality_probs and modality_probs[0][1] > confidence:
                            logger.info(f"Using modality-filtered prediction: {display_names[modality_probs[0][0]]} ({modality_probs[0][1]:.4f})")
                            pred_idx = modality_probs[0][0]
                            pred_class_dataset = class_names[pred_idx]
                            pred_class_display = display_names[pred_idx]
                            confidence = modality_probs[0][1]
            else:
                # Auto modality - trust the model prediction
                logger.info(f"Auto modality detected: {modality}, model predicted: {pred_mod}, confidence: {confidence:.4f}")
            # Use display name for UI, but normalize for medical database lookup
            display = pred_class_display if confidence >= threshold else f"{pred_class_display} (Low Confidence: {confidence:.1%})"
            normalized_key = normalize_class_key(pred_class_display)
            medical_info = MEDICAL_DATABASE.get(normalized_key, {})
            return {
                "predicted_class": display,
                "base_class_name": pred_class_display,  # Use display name
                "confidence": confidence,
                "all_probabilities": all_probs,
                "class_names": display_names,  # Return display names for UI
                "medical_info": medical_info
            }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        return {
            "predicted_class":"Error",
            "base_class_name":"Error",
            "confidence":0.0,
            "all_probabilities":[0.0]*len(CLASS_NAMES),
            "class_names":CLASS_NAMES,
            "error": str(e)
        }

# -------------------- Grad-CAM++ visualization --------------------
def create_gradcam_plus_plus_overlay(image: Image.Image, predicted_class: str) -> Image.Image:
    try:
        model = load_model()
        # modality used only for preprocessing
        modality = infer_modality_from_class(predicted_class)
        input_t = preprocess_image(image, modality=modality)
        # find a target_conv layer: search in model (matching train.py architecture)
        target_layer = None
        model_candidate = model
        # search in model.model.features if exists (EfficientNet from train.py)
        if hasattr(model_candidate, "model") and hasattr(model_candidate.model, "features"):
            for m in reversed(list(model_candidate.model.features.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
        # search in model.base if exists (ResNet)
        if target_layer is None and hasattr(model_candidate, "base"):
            for m in reversed(list(model_candidate.base.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
        # fallback: search in model.backbone if exists (old architecture)
        if target_layer is None and hasattr(model_candidate, "backbone"):
            for m in reversed(list(model_candidate.backbone.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
        # final fallback: search entire model
        if target_layer is None:
            for m in reversed(list(model_candidate.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
        if target_layer is None:
            logger.warning("No conv layer found for Grad-CAM++ - returning original image")
            return image
        gradpp = GradCAMPlusPlus(model_candidate, target_layer)
        cam_mask = gradpp.generate(input_t)
        # overlay
        orig = np.array(image.convert("RGB"))
        h,w = orig.shape[:2]
        cam_resized = cv2.resize(cam_mask, (w,h))
        cam_uint8 = np.uint8(255 * cam_resized)
        colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        alpha = 0.55
        overlay = cv2.addWeighted(orig.astype(np.uint8), 1.0 - alpha, colored.astype(np.uint8), alpha, 0)
        # draw contour
        try:
            contours, _ = cv2.findContours(cam_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255,255,255), 2)
        except Exception:
            pass
        return Image.fromarray(overlay)
    except Exception as e:
        logger.error(f"Grad-CAM++ error: {e}")
        traceback.print_exc()
        return image

# -------------------- Hospital lookup (kept original logic improved) --------------------
def geocode_location(location: str) -> Optional[Dict]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location, "format":"json", "limit":1}
        headers = {"User-Agent":"MedicalAI-App/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        res = r.json()
        if res:
            return {"lat": float(res[0]["lat"]), "lng": float(res[0]["lon"]), "address": res[0].get("display_name", location)}
        return None
    except Exception as e:
        logger.warning(f"Geocode failed: {e}")
        return None

def get_realistic_hospitals_by_coordinates(lat: float, lng: float, condition: str) -> List[Dict]:
    # same list as before, trimmed for brevity
    hospital_networks = {
        (19.0,72.8): [{"name":"Lilavati Hospital","address":"Bandra West, Mumbai","phone":"+91-22-66666666","rating":4.5,"specialty":"Multi-specialty"}],
        (28.6,77.2): [{"name":"Apollo Hospitals Delhi","address":"Sarita Vihar, New Delhi","phone":"+91-11-29871000","rating":4.5,"specialty":"Multi-specialty"}],
        (12.9,77.6): [{"name":"Apollo Hospitals Bangalore","address":"Bannerghatta Road, Bangalore","phone":"+91-80-26304050","rating":4.5,"specialty":"Multi-specialty"}],
        (17.3850,78.4867): [{"name":"Apollo Hospitals Hyderabad","address":"Jubilee Hills, Hyderabad","phone":"+91-40-23607777","rating":4.5,"specialty":"Multi-specialty"}]
    }
    min_d = float("inf")
    chosen = None
    for (cy,cx), hs in hospital_networks.items():
        d = math.hypot(lat-cy, lng-cx)
        if d < min_d:
            min_d = d
            chosen = hs
    if not chosen:
        return []
    out = []
    for h in chosen[:5]:
        out.append({"name":h["name"], "address":h["address"], "phone":h["phone"], "rating":h["rating"], "specialty":h["specialty"], "distance":f"{(min_d*111):.1f} km"})
    return out

def get_hospitals_near_location(location: str, condition: str) -> List[Dict]:
    try:
        geo = geocode_location(location)
        if not geo:
            return []
        return get_realistic_hospitals_by_coordinates(geo["lat"], geo["lng"], condition)
    except Exception as e:
        logger.error(f"Hospital search error: {e}")
        return []

def create_hospital_map(location: str, hospitals: List[Dict]) -> str:
    try:
        center_lat, center_lng = 17.3850,78.4867
        geo = geocode_location(location)
        if geo:
            center_lat, center_lng = geo["lat"], geo["lng"]
        m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")
        for i,h in enumerate(hospitals):
            folium.Marker(location=[center_lat + (i*0.01), center_lng + (i*0.01)], popup=f"{h['name']}<br>{h['address']}<br>Phone: {h['phone']}<br>Distance: {h['distance']}").add_to(m)
        return m._repr_html_()
    except Exception as e:
        logger.error(f"Map creation error: {e}")
        return "<p>Map unavailable</p>"

# -------------------- Report generation --------------------
def generate_medical_report(prediction_result: Dict, image: Image.Image, highlighted_image: Image.Image) -> str:
    try:
        medical_info = prediction_result.get("medical_info", {})
        predicted_class = prediction_result.get("predicted_class", "Unknown")
        confidence = prediction_result.get("confidence", 0.0)
        report = f"""
        <div style="font-family:Arial,sans-serif; max-width:800px; margin:0 auto; padding:20px;">
          <h1 style="text-align:center; color:#2c3e50;">Medical Analysis Report</h1>
          <hr style="border:2px solid #3498db;">
          <div style="background:#f8f9fa; padding:15px; border-radius:10px;">
            <h3>Diagnosis Summary</h3>
            <p><strong>Condition:</strong> {medical_info.get('condition', predicted_class)}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Severity:</strong> {medical_info.get('severity','Unknown')}</p>
            <p><strong>Urgency:</strong> {medical_info.get('urgency','Unknown')}</p>
          </div>
          <div style="margin-top:12px; background:#e8f5e8; padding:12px; border-radius:8px;">
            <h4>Description</h4>
            <p>{medical_info.get('description','No description available.')}</p>
          </div>
        </div>
        """
        return report
    except Exception as e:
        logger.error(f"Report gen error: {e}")
        return "<p>Report generation failed</p>"

# -------------------- UI: Gradio interface with floating SOS and animations -------------
CUSTOM_CSS = """
/* Basic theming */
.gradio-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height:100vh; padding-bottom:60px; }
.main-header { text-align:center; color:white; padding:30px; border-radius:12px; margin-bottom:12px; }
.feature-card { background:white; border-radius:12px; padding:14px; margin:8px; box-shadow:0 6px 18px rgba(0,0,0,0.12); }
.emergency-button { background: linear-gradient(45deg,#ff4444,#cc0000); color:white; border:none; padding:12px 18px; border-radius:999px; font-weight:700; }
.sos-floating { position: fixed; right: 24px; bottom: 24px; z-index: 9999; }
/* loading spinner */
.loading-overlay { position: absolute; inset:0; display:flex; align-items:center; justify-content:center; background: rgba(255,255,255,0.6); border-radius:12px; }
"""

def create_enhanced_interface():
    with gr.Blocks(title="🩺 Enhanced Medical AI Platform", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        # Header
        gr.HTML("""<div class="main-header"><h1>🩺 Enhanced Medical AI Platform</h1>
                   <p style="margin-top:6px;">AI-assisted medical image analysis — brain/chest/bone</p></div>""")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Upload Medical Image")
                image_input = gr.Image(label="Upload (X-ray / MRI)", type="pil", elem_classes=["feature-card"], height=360)
                gr.Markdown("### 📍 Your Location")
                location_input = gr.Textbox(label="City / Area (optional, for hospitals)", placeholder="e.g., Hyderabad, India", elem_classes=["feature-card"])
                gr.Markdown("### ⚙️ Analysis Settings")
                modality_input = gr.Radio(choices=[("🧠 Brain MRI","brain"),("🫁 Chest X-ray","chest"),("🦴 Bone X-ray","bone"),("🤖 Auto-detect","auto")], value="auto", label="Image Type (Select for best accuracy)", elem_classes=["feature-card"])
                threshold_slider = gr.Slider(minimum=0.0, maximum=0.9, value=0.25, step=0.05, label="Confidence Threshold", elem_classes=["feature-card"])
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", elem_classes=["emergency-button"])
                gr.HTML("""<div style="margin-top:10px; text-align:center;">
                              <button class="emergency-button" onclick="window.open('tel:108')">🚑 Ambulance: 108</button>
                              <button class="emergency-button" onclick="window.open('tel:100')">👮 Police: 100</button>
                           </div>""")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Diagnosis Results"):
                        diagnosis_output = gr.HTML(value="<div style='text-align:center; padding:40px; color:#eee;'>Upload an image and click Analyze</div>")
                    with gr.Tab("🔍 Affected Areas"):
                        gradcam_display = gr.Image(type="pil", label="Grad-CAM++ Overlay", height=540)
                    with gr.Tab("📈 Confidence Analysis"):
                        confidence_output = gr.HTML(value="<div style='text-align:center; padding:40px; color:#eee;'>Confidence bars will appear here</div>")
                    with gr.Tab("📋 Detailed Report"):
                        report_output = gr.HTML(value="<div style='text-align:center; padding:40px; color:#eee;'>Detailed medical report will appear here</div>")
                    with gr.Tab("🏥 Nearby Hospitals"):
                        hospitals_output = gr.HTML(value="<div style='text-align:center; padding:40px; color:#eee;'>Hospital map will appear here</div>")

        # Floating SOS (outside normal flow)
        gr.HTML("""<div class="sos-floating"><button class="emergency-button" onclick="window.open('tel:108')">🚨 SOS</button></div>""")

        # processing function
        def process(image, location, threshold, modality):
            if image is None:
                return ("<div style='text-align:center;color:#fff;'>Please upload an image.</div>", None, "<div style='text-align:center;color:#fff;'>No data</div>", "<div style='text-align:center;color:#fff;'>No report</div>", "<div style='text-align:center;color:#fff;'>No hospitals</div>")
            modality_hint = None if modality == "auto" else modality
            diagnosis, highlighted, confidence_html, report, hospitals_html = run_analysis_and_render(image, location, threshold, modality_hint)
            return diagnosis, highlighted, confidence_html, report, hospitals_html

        analyze_btn.click(fn=process, inputs=[image_input, location_input, threshold_slider, modality_input], outputs=[diagnosis_output, gradcam_display, confidence_output, report_output, hospitals_output])
        image_input.change(fn=process, inputs=[image_input, location_input, threshold_slider, modality_input], outputs=[diagnosis_output, gradcam_display, confidence_output, report_output, hospitals_output])

    return demo

# -------------------- High-level orchestrator (used by UI) --------------------
def run_analysis_and_render(image: Image.Image, location: str, threshold: float, modality_hint: Optional[str]):
    # Step 1: Predict
    pred = predict_medical_image(image, threshold, modality_hint)
    if "error" in pred:
        diagnosis_html = f"<div style='color:#fff;'>{pred.get('error')}</div>"
        return diagnosis_html, image, "<div style='color:#fff;'>Prediction failed</div>", "<div>Report failed</div>", "<div>No hospitals</div>"
    # Step 2: Grad-CAM++ overlay (use base_class_name for precise feature extraction)
    base = pred.get("base_class_name", pred["predicted_class"])
    highlighted = create_gradcam_plus_plus_overlay(image, base)
    # Step 3: Prepare diagnosis card
    medical_info = pred.get("medical_info", {})
    cond = medical_info.get("condition", pred["predicted_class"])
    confidence = pred.get("confidence", 0.0)
    diagnosis_html = f"""
    <div style="font-family:Arial;color:#fff;padding:16px;">
      <h3 style="margin:6px 0 6px 0;">🔍 Diagnosis</h3>
      <p><strong>Condition:</strong> {cond}</p>
      <p><strong>Confidence:</strong> {confidence:.1%}</p>
      <p><strong>Advice:</strong> {medical_info.get('treatment','Consult a specialist')}</p>
    </div>
    """
    # Step 4: Confidence bars html
    class_names = pred.get("class_names", [])
    probs = pred.get("all_probabilities", [])
    bars_html = "<div style='font-family:Arial;padding:10px;max-width:700px;'>"
    for cname, p in zip(class_names, probs):
        color = "#4CAF50" if p>0.5 else "#FFA500" if p>0.2 else "#F44336"
        bars_html += f"<div style='background:{color}; color:white; padding:6px; margin:6px 0; border-radius:6px;'>{cname}: {p:.1%}</div>"
    bars_html += "</div>"
    # Step 5: Report
    report_html = generate_medical_report(pred, image, highlighted)
    # Step 6: Hospitals
    hospitals_html = ""
    if location and location.strip():
        hospitals = get_hospitals_near_location(location, base)
        hospitals_html = create_hospital_map(location, hospitals)
    return diagnosis_html, highlighted, bars_html, report_html, hospitals_html

# -------------------- Main --------------------
def main():
    # ensure model file exists warning
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model path {MODEL_PATH} not found. Place your model at that path before running.")
    interface = create_enhanced_interface()
    interface.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=False)

if __name__ == "__main__":
    main()
