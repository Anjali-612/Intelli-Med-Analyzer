# train.py  (Windows-safe, no spawn error, EfficientNet, saves unified checkpoint)
import os
import json
import random
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import ImageFile

# ---- handle truncated images ----
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

DATA_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

IMG_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 16         # lower to 8 if OOM
EPOCHS = 15
LR = 1e-4
VAL_FRACTION = 0.10
SEED = 42
CHECKPOINT_PATH = "medical_model.pth"

# ---- choose num_workers safely ----
# On Windows use 0 to avoid multiprocessing spawn problems. Increase on Linux.
NUM_WORKERS = 0 if os.name == "nt" else 4

# ---- transforms ----
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---- helper: build model ----
def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ---- training / validation routines ----
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{len(loader)} - loss: {loss.item():.4f}")

    avg_loss = running_loss / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

# ---- main entrypoint ----
def main():
    try:
        torch.manual_seed(SEED)
        random.seed(SEED)

        if not os.path.isdir(TRAIN_DIR):
            raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

        # Full dataset using train transforms initially
        full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
        class_names = full_dataset.classes  # Actual dataset folder names
        class_to_idx = full_dataset.class_to_idx
        print("Detected classes (dataset folder names):", class_names)
        print("Total images (train):", len(full_dataset))

        # Mapping from dataset folder names to display names (for UI/medical database)
        DISPLAY_NAME_MAPPING = {
            "NORMAL": "chest_xray/NORMAL",
            "PNEUMONIA": "chest_xray/PNEUMONIA",
            "fractured": "bone_fracture/fractured",
            "not fractured": "bone_fracture/not fractured",
            "glioma": "brain_tumor/glioma",
            "meningioma": "brain_tumor/meningioma",
            "notumor": "brain_tumor/notumor",
            "pituitary": "brain_tumor/pituitary"
        }
        
        # Create display names list in the same order as class_names
        display_names = [DISPLAY_NAME_MAPPING.get(name, name) for name in class_names]
        print("Display names (for app):", display_names)

        # Save actual class names (dataset folder names)
        with open("class_names.json", "w") as fh:
            json.dump(class_names, fh)

        with open("class_to_idx.json", "w") as fh:
            json.dump(class_to_idx, fh)
        
        # Save display names mapping
        with open("display_names.json", "w") as fh:
            json.dump(display_names, fh)
        
        with open("name_mapping.json", "w") as fh:
            json.dump(DISPLAY_NAME_MAPPING, fh)

        # Create train/val split
        val_size = int(VAL_FRACTION * len(full_dataset))
        val_size = max(1, val_size)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(SEED))
        # Ensure val uses val transforms
        val_ds.dataset.transform = val_tfms

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))

        # Build model
        model = build_model(len(class_names)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2, verbose=True)

        best_val_acc = 0.0

        print("\nStarting training loop...")
        for epoch in range(EPOCHS):
            t0 = time.time()
            print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1} finished in {time.time()-t0:.1f}s")
            print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
            print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": class_names,  # Actual dataset folder names
                    "display_names": display_names,  # Display names for UI
                    "class_to_idx": class_to_idx,
                    "name_mapping": DISPLAY_NAME_MAPPING,  # Mapping dict
                    "norm": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}
                }, CHECKPOINT_PATH)
                print(f"  → New best model saved (val_acc={best_val_acc:.2f}%) to {CHECKPOINT_PATH}")

        print("\nTraining complete. Best val acc: {:.2f}%".format(best_val_acc))

        # OPTIONAL: evaluate on test set if present
        if os.path.isdir(TEST_DIR) and any(os.scandir(TEST_DIR)):
            print("\nEvaluating on test set...")
            test_ds = datasets.ImageFolder(TEST_DIR, transform=val_tfms)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        print("All done. Model & metadata saved as:", CHECKPOINT_PATH)

    except Exception as e:
        print("Fatal error in training loop:")
        traceback.print_exc()

# Standard Windows-safe guard
if __name__ == "__main__":
    # required for Windows frozen executables and safe spawn
    if os.name == "nt":
        import torch.multiprocessing
        torch.multiprocessing.freeze_support()
    main()
