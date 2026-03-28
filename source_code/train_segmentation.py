import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Paths
DATA_PATH = "data/processed/segmentation/train"
MODEL_SAVE_PATH = "models/segmentation.pth"

IMG_SIZE = 256
EPOCHS = 3
BATCH_SIZE = 8


# -------------------------------
# Load Data
# -------------------------------
def load_data():
    print("Loading segmentation data...")

    images = []
    masks = []

    img_dir = os.path.join(DATA_PATH, "images")
    mask_dir = os.path.join(DATA_PATH, "masks")

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        image = image / 255.0
        mask = mask / 255.0

        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


# -------------------------------
# Prepare Data
# -------------------------------
def prepare_data(images, masks):
    images = np.transpose(images, (0, 3, 1, 2))  # NCHW
    masks = np.expand_dims(masks, axis=1)        # add channel

    X = torch.tensor(images, dtype=torch.float32)
    y = torch.tensor(masks, dtype=torch.float32)

    return X, y


# -------------------------------
# Model
# -------------------------------
def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


# -------------------------------
# Dice Loss
# -------------------------------
def dice_loss(pred, target):
    smooth = 1e-6

    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2. * intersection + smooth) / (union + smooth)


# -------------------------------
# Train
# -------------------------------
def train_model(model, X, y):
    print("Training segmentation model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dataset_size = X.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for i in range(0, dataset_size, BATCH_SIZE):
            batch_X = X[i:i+BATCH_SIZE].to(device)
            batch_y = y[i:i+BATCH_SIZE].to(device)

            outputs = model(batch_X)

            loss = dice_loss(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    return model


# -------------------------------
# Save Model
# -------------------------------
def save_model(model):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Segmentation model saved!")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    images, masks = load_data()
    X, y = prepare_data(images, masks)

    model = build_model()
    model = train_model(model, X, y)

    save_model(model)