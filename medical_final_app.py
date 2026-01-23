import os
import shutil

BASE = "dataset"   # change if needed

CLASSES = {
    "bone_fracture": ["fractured", "not fractured"],
    "brain_tumor": ["glioma", "meningioma", "notumor", "pituitary"],
    "chest_xray": ["NORMAL", "PNEUMONIA"]
}

splits = ["train", "val", "test"]

for split in splits:
    print("\nProcessing", split)

    split_path = os.path.join(BASE, split)

    for top_folder, sub_classes in CLASSES.items():
        for sub in sub_classes:
            src = os.path.join(split_path, top_folder, sub)
            dst = os.path.join(split_path, sub)

            os.makedirs(dst, exist_ok=True)

            if not os.path.exists(src):
                continue

            for file in os.listdir(src):
                file_path = os.path.join(src, file)
                if os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(dst, file))

    # Remove empty top-level folders
    for top_folder in CLASSES.keys():
        folder_path = os.path.join(split_path, top_folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

print("\n✔ Dataset successfully restructured into 8 FLAT classes!")
