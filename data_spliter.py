import os
import random
from pathlib import Path

random.seed(42)

DATASET_ROOT = Path(
    "pneumonia_dataset/Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)/Curated X-Ray Dataset/Curated X-Ray Dataset"
)

OUTPUT_DIR = Path("data_splits")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

label_map = {
    "Normal": 0,
    "Pneumonia-Bacterial": 1,
    "Pneumonia-Viral": 2,
    "COVID-19": 3
}

split_files = {
    "train": [],
    "val": [],
    "test": []
}

for class_name, label in label_map.items():
    class_dir = DATASET_ROOT / class_name
    images = list(class_dir.glob("*"))

    random.shuffle(images)
    n = len(images)

    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        for img_path in files:
            split_files[split].append(
                f"{img_path.resolve()},{label}\n"
            )

for split, lines in split_files.items():
    with open(OUTPUT_DIR / f"{split}.txt", "w") as f:
        f.writelines(lines)

print("Dataset splitting complete")