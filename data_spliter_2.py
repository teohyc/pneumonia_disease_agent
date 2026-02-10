import os
import random
from pathlib import Path

random.seed(42)

DATASET_ROOT = Path(
    "pneumonia_dataset/Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)/Curated X-Ray Dataset/Curated X-Ray Dataset"
)

GENERATED_COVID_DIR = Path("covid_only_generated")

OUTPUT_DIR = Path("data_splits_2")
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

#process generated COVID-19 images
if GENERATED_COVID_DIR.exists():
    generated_images = list(GENERATED_COVID_DIR.glob("*"))
    random.shuffle(generated_images)
    n = len(generated_images)

    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    SPLITS = {
        "train": generated_images[:n_train],
        "val": generated_images[n_train:n_train + n_val],
        "test": generated_images[n_train + n_val:]
    }

    for split, images in SPLITS.items():
        for img_path in images:
            split_files[split].append(
                f"{img_path.resolve()},{label_map['COVID-19']}\n"
            )

for split, lines in split_files.items():
    with open(OUTPUT_DIR / f"{split}.txt", "w") as f:
        f.writelines(lines)

print("Dataset splitting complete")