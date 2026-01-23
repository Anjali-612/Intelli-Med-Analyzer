#!/usr/bin/env python3
"""
Enhanced Medical Image Training with Improved Accuracy
- Better data augmentation
- Improved model architecture
- Enhanced training techniques
- Better validation and testing
"""

import os
import logging
from pathlib import Path
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from collections import Counter
import json
import numpy as np
import random

# Fix truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")

class EnhancedResNet(nn.Module):
    """Enhanced ResNet with attention mechanisms and better feature extraction"""
    
    def __init__(self, num_classes, pretrained=True):
        super(EnhancedResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add attention modules
        self.attention1 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced classifier
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
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention1(features)
        attended_features = features * attention_weights
        
        # Global pooling
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, attended_features

class MedicalDataset(Dataset):
    def __init__(self, root, transform=None, max_samples=2000):
        self.samples = []
        self.transform = transform
        root = Path(root)
        
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")

        # Find all images with better sampling
        class_counts = Counter()
        for img_path in root.rglob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # Check path length to avoid Windows issues
                if len(str(img_path)) > 200:
                    logger.warning(f"Skipping long path: {img_path}")
                    continue
                    
                parts = img_path.parts
                if len(parts) >= 3:
                    domain = parts[-3]
                    class_name = parts[-2]
                    label = f"{domain}_{class_name}"
                    
                    if class_counts[label] < max_samples // 8:  # Limit per class
                        self.samples.append((str(img_path), label))
                        class_counts[label] += 1

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

        self.classes = sorted(list(set([label for _, label in self.samples])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        logger.info(f"Found {len(self.samples)} samples across {len(self.classes)} classes")
        logger.info(f"Class distribution: {dict(class_counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Use shorter path handling for Windows
            if len(path) > 200:
                # Copy to temp location with shorter path
                import tempfile
                import shutil
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, f"img_{idx}.jpg")
                shutil.copy2(path, temp_file)
                image = Image.open(temp_file).convert("RGB")
                # Clean up temp file
                try:
                    os.remove(temp_file)
                    os.rmdir(temp_dir)
                except:
                    pass
            else:
                image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Skipping corrupted image: {path}")
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]

def get_enhanced_transforms(img_size=224, mode='train'):
    """Enhanced data augmentation for better generalization"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 64, img_size + 64)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def focal_loss(pred, target, alpha=1, gamma=2):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def label_smoothing_loss(pred, target, smoothing=0.1):
    """Label smoothing for better generalization"""
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(pred, dim=1)
    nll_loss = F.nll_loss(logprobs, target, reduction='none')
    smooth_loss = -logprobs.mean(dim=1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def train_model(model, criterion, optimizer, dataloaders, num_epochs=20, scheduler=None):
    """Enhanced training with better monitoring and techniques"""
    best_acc = 0.0
    best_model_state = None
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, _ = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    # Use different loss functions for training and validation
                    if phase == "train":
                        # Combine focal loss and label smoothing
                        loss1 = focal_loss(outputs, labels)
                        loss2 = label_smoothing_loss(outputs, labels)
                        loss = 0.7 * loss1 + 0.3 * loss2
                    else:
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc.item())
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = model.state_dict().copy()
                logger.info(f"New best validation accuracy: {best_acc:.4f}")
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation accuracy: {best_acc:.4f}")
    
    return model, train_losses, val_accuracies

def main():
    """Enhanced main training function"""
    logger.info("Starting enhanced medical image training...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Paths
    train_dir = "dataset/train"
    val_dir = "dataset/val"
    
    # Create datasets with more samples for better training
    train_dataset = MedicalDataset(train_dir, transform=get_enhanced_transforms(mode='train'), max_samples=2000)
    val_dataset = MedicalDataset(val_dir, transform=get_enhanced_transforms(mode='val'), max_samples=500)
    
    # Create data loaders with better parameters
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    }
    
    # Initialize enhanced model
    model = EnhancedResNet(len(train_dataset.classes), pretrained=True)
    model = model.to(device)
    
    # Enhanced loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Train model with more epochs
    trained_model, train_losses, val_accuracies = train_model(
        model, criterion, optimizer, dataloaders, num_epochs=20, scheduler=scheduler
    )
    
    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = "model/best_model.pth"
    torch.save(trained_model.state_dict(), model_path)
    
    # Save class map
    class_map = {
        "class_names": train_dataset.classes,
        "img_size": 224,
        "num_classes": len(train_dataset.classes),
        "model_type": "EnhancedResNet"
    }
    with open("model/class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    
    # Save training history
    training_history = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "best_accuracy": max(val_accuracies) if val_accuracies else 0
    }
    with open("model/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Class map saved to model/class_map.json")
    logger.info(f"Training history saved to model/training_history.json")
    
    # Quick test
    trained_model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attention_maps = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(min(3, inputs.size(0))):
                true_class = train_dataset.classes[labels[j]]
                pred_class = train_dataset.classes[preds[j]]
                confidence = F.softmax(outputs[j], dim=0)[preds[j]].item()
                print(f"Sample {i*16 + j + 1}: True={true_class}, Pred={pred_class} (Conf: {confidence:.3f})")
            break
    
    logger.info("Training completed successfully!")
    logger.info("🚀 You can now run: python medical_platform_enhanced.py")

if __name__ == "__main__":
    main()














