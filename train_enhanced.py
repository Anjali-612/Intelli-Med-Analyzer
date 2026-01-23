# train.py (robust — works with or without dataset/val)
import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile

# --------------------------
# CRITICAL FIXES FOR WINDOWS / CORRUPTED IMAGES
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# --------------------------
# SETTINGS / HYPERPARAMETERS
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")       # may or may not exist
TEST_DIR = os.path.join(DATASET_DIR, "test")

batch_size = 16
epochs = 15
learning_rate = 1e-4
img_size = 224
val_split = 0.10   # fraction to use for validation if VAL_DIR missing
random_seed = 42
num_workers = 0    # set to 0 on Windows; increase on Linux

# --------------------------
# TRANSFORMS
# --------------------------
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------
# LOAD DATASETS
# --------------------------
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

# If a VAL_DIR exists and has files, use it; otherwise we'll split train_dataset
use_external_val = os.path.isdir(VAL_DIR) and any(os.scandir(VAL_DIR))
if use_external_val:
    print("Detected external validation directory:", VAL_DIR)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=test_transform)
else:
    print("No external validation directory detected. Will split training set for validation.")
    val_dataset = None

# Test dataset (required)
if not os.path.isdir(TEST_DIR):
    print("WARNING: Test directory not found:", TEST_DIR)
    test_dataset = None
else:
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# Get class names from train dataset (ImageFolder enforces same classes usually)
class_names = train_dataset.classes
print("Detected Classes:", class_names)

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# --------------------------
# CREATE DATLoaders (including optional split)
# --------------------------
if val_dataset is None:
    # Create reproducible split
    total = len(train_dataset)
    indices = list(range(total))
    random.seed(random_seed)
    random.shuffle(indices)
    val_size = max(1, int(math.floor(val_split * total)))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=test_transform), val_indices)  # val gets test_transform
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Split train -> train:{len(train_indices)} val:{len(val_indices)}")
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Using external val dataset with {len(val_dataset)} samples")

if test_dataset is not None:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Test dataset has {len(test_dataset)} samples")
else:
    test_loader = None
    print("No test dataset provided; evaluation on test set will be skipped.")

# --------------------------
# BUILD EfficientNet-B0 MODEL
# --------------------------
# We use pre-trained weights as you did in training (can set weights=None if you don't want imagenet)
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Replace classifier head to match number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))
model = model.to(DEVICE)

# --------------------------
# LOSS & OPTIMIZER
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.3, verbose=True)

# --------------------------
# TRAIN/VAL FUNCTIONS
# --------------------------
def train_one_epoch(epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch_idx+1}/{epochs}")
    print("-" * 40)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    return train_loss, train_acc

def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    val_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
    return val_loss, val_acc

# --------------------------
# TRAINING LOOP
# --------------------------
best_val_acc = 0.0
best_model_path = "best_model_EfficientNetB0.pth"

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = validate()

    # Scheduler expects a scalar metric; we provide val_loss
    scheduler.step(val_loss)

    # Save best model by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✔ New best model saved with val acc: {best_val_acc:.2f}%")

print("\nTraining loop complete.")

# If we never saved a best model (edge case), save final
if not os.path.exists(best_model_path):
    torch.save(model.state_dict(), best_model_path)
    print("Saved final model as best_model_EfficientNetB0.pth")

# --------------------------
# TESTING (if test set exists)
# --------------------------
if test_loader is not None:
    print("\nEvaluating on TEST set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(all_labels, all_preds))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n✔ Model saved as", best_model_path)
    print("✔ Class names saved as class_names.json")
else:
    print("\nNo test set available; skipping test evaluation.")
