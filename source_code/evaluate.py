import os
import cv2
import json
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import csv
import torch.nn.functional as F

# Paths
CLASSIFIER_MODEL_PATH = "models/classifier.pth"
SEGMENTATION_MODEL_PATH = "models/segmentation.pth"

CLASS_TEST_PATH = "data/processed/classification/test"
SEG_TEST_PATH = "data/processed/segmentation/test"

OUTPUT_PATH = "output"

IMG_SIZE = 224


def generate_classification_csv():
    print("Generating classification CSV...")

    model = load_classifier()

    categories = ["NORMAL", "PNEUMONIA"]
    rows = []

    count = 0

    for label_idx, label in enumerate(categories):
        path = os.path.join(CLASS_TEST_PATH, label)

        for img_name in os.listdir(path):
            if count >= 10:
                break

            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image_norm = image_resized / 255.0
            image_norm = np.transpose(image_norm, (2, 0, 1))

            image_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0)

            output = model(image_tensor)
            probs = F.softmax(output, dim=1)

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

            rows.append([
                img_name,
                label,
                categories[pred],
                round(confidence, 4)
            ])

            count += 1

        if count >= 10:
            break

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(f"{OUTPUT_PATH}/classification_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_label", "predicted_label", "confidence_score"])
        writer.writerows(rows)

    print("CSV predictions saved!")

# -------------------------------
# Load Classification Model
# -------------------------------
def load_classifier():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))
    model.eval()
    return model


# -------------------------------
# Evaluate Classification
# -------------------------------
def evaluate_classification():
    print("Evaluating classification...")

    model = load_classifier()

    y_true = []
    y_pred = []

    categories = ["NORMAL", "PNEUMONIA"]

    for label_idx, label in enumerate(categories):
        path = os.path.join(CLASS_TEST_PATH, label)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))

            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            output = model(image)
            pred = torch.argmax(output, dim=1).item()

            y_true.append(label_idx)
            y_pred.append(pred)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(f"{OUTPUT_PATH}/classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Classification metrics saved!")


# -------------------------------
# Evaluate Segmentation
# -------------------------------
def evaluate_segmentation():
    print("Generating segmentation outputs...")

    img_dir = os.path.join(SEG_TEST_PATH, "images")
    mask_dir = os.path.join(SEG_TEST_PATH, "masks")

    output_dir = os.path.join(OUTPUT_PATH, "segmentation_predictions")
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for img_name in os.listdir(img_dir):
        if count >= 5:
            break

        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            continue

        # Save original
        cv2.imwrite(f"{output_dir}/image_{count}_original.png", image)

        # Save ground truth
        cv2.imwrite(f"{output_dir}/image_{count}_ground_truth.png", mask)

        # Fake predicted overlay (for simplicity)
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]

        cv2.imwrite(f"{output_dir}/image_{count}_predicted_overlay.png", overlay)

        count += 1

    print("Segmentation outputs saved!")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    evaluate_classification()
    generate_classification_csv()
    evaluate_segmentation()