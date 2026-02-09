import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "pneumonia_dataset/Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)/Curated X-Ray Dataset/Curated X-Ray Dataset/COVID-19"
TARGET_DIR = "covid_only_train"

os.makedirs(TARGET_DIR, exist_ok=True)

# Filter for actual image files (exclude directories)
all_files = os.listdir(SOURCE_DIR)
images = [f for f in all_files if os.path.isfile(os.path.join(SOURCE_DIR, f))]

if len(images) == 0:
    print(f"Error: No image files found in {SOURCE_DIR}")
    print(f"Contents: {all_files}")
else:
    train_imgs, _ = train_test_split(images, train_size=0.8, random_state=42)

    for img in train_imgs: #extract only training images
        shutil.copy(
            os.path.join(SOURCE_DIR, img),
            os.path.join(TARGET_DIR, img)
        )

print(f"Copied {len(train_imgs)} images to {TARGET_DIR}")