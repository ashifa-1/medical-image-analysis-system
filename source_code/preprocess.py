import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Paths
RAW_XRAY_PATH = "data/raw/chest_xray"
RAW_MRI_PATH = "data/raw/lgg-mri-segmentation"

PROCESSED_PATH = "data/processed"

IMG_SIZE_CLASSIFICATION = 224
IMG_SIZE_SEGMENTATION = 256


# Create folders
def create_dirs():
    for task in ["classification", "segmentation"]:
        for split in ["train", "val", "test"]:
            os.makedirs(f"{PROCESSED_PATH}/{task}/{split}", exist_ok=True)

# Preprocess Classification Data
def preprocess_classification():
    print("Processing Chest X-ray Dataset...")

    categories = ["NORMAL", "PNEUMONIA"]
    data = []
    labels = []

    for category in categories:
        path = os.path.join(RAW_XRAY_PATH, "train", category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path)

            # Skip corrupted / non-image files
            if image is None:
                print(f"Skipping corrupted image: {img_path}")
                continue

            image = cv2.resize(image, (IMG_SIZE_CLASSIFICATION, IMG_SIZE_CLASSIFICATION))
            image = image / 255.0

            data.append(image)
            labels.append(category)

    data = np.array(data)
    labels = np.array(labels)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    save_classification_split(X_train, y_train, "train")
    save_classification_split(X_val, y_val, "val")
    save_classification_split(X_test, y_test, "test")


def save_classification_split(images, labels, split):
    for i in range(len(images)):
        label = labels[i]
        save_dir = f"{PROCESSED_PATH}/classification/{split}/{label}"
        os.makedirs(save_dir, exist_ok=True)

        img = (images[i] * 255).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/{i}.png", img)


# Preprocess MRI Segmentation Data
def preprocess_segmentation():
    print("Processing MRI Dataset...")

    images = []
    masks = []

    for folder in tqdm(os.listdir(RAW_MRI_PATH)):
        folder_path = os.path.join(RAW_MRI_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        image_path = None
        mask_path = None

        # Find image and mask safely
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if "mask" in file.lower():
                mask_path = file_path
            else:
                image_path = file_path

        # Skip incomplete folders
        if image_path is None or mask_path is None:
            print(f"Skipping incomplete folder: {folder_path}")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)

        # Skip bad files
        if image is None or mask is None:
            print(f"Skipping bad file in {folder_path}")
            continue

        image = cv2.resize(image, (IMG_SIZE_SEGMENTATION, IMG_SIZE_SEGMENTATION))
        mask = cv2.resize(mask, (IMG_SIZE_SEGMENTATION, IMG_SIZE_SEGMENTATION))

        image = image / 255.0
        mask = mask / 255.0

        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    save_segmentation_split(X_train, y_train, "train")
    save_segmentation_split(X_val, y_val, "val")
    save_segmentation_split(X_test, y_test, "test")


def save_segmentation_split(images, masks, split):
    for i in range(len(images)):
        img_dir = f"{PROCESSED_PATH}/segmentation/{split}/images"
        mask_dir = f"{PROCESSED_PATH}/segmentation/{split}/masks"

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        img = (images[i] * 255).astype(np.uint8)
        mask = (masks[i] * 255).astype(np.uint8)

        cv2.imwrite(f"{img_dir}/{i}.png", img)
        cv2.imwrite(f"{mask_dir}/{i}.png", mask)


# MAIN
if __name__ == "__main__":
    create_dirs()
    preprocess_classification()
    preprocess_segmentation()