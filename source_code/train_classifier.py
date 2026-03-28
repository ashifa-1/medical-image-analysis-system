import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = "data/processed/classification/train"
MODEL_SAVE_PATH = "models/classifier.pth"

IMG_SIZE = 224
EPOCHS = 3
BATCH_SIZE = 16


# -------------------------------
# Load Data
# -------------------------------
def load_data():
    print("Loading classification data...")

    images = []
    labels = []

    categories = ["NORMAL", "PNEUMONIA"]

    for label in categories:
        path = os.path.join(DATA_PATH, label)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# -------------------------------
# Prepare Data
# -------------------------------
def prepare_data(images, labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    images = np.transpose(images, (0, 3, 1, 2))

    X = torch.tensor(images, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


# -------------------------------
# Model
# -------------------------------
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model


# -------------------------------
# Train
# -------------------------------
def train_model(model, X, y):
    print("Training classifier...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

    dataset_size = X.shape[0]

    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        # Manual batching
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_X = X[i:i+BATCH_SIZE].to(device)
            batch_y = y[i:i+BATCH_SIZE].to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

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
    print("Model saved at:", MODEL_SAVE_PATH)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    images, labels = load_data()
    X, y = prepare_data(images, labels)

    model = build_model()
    model = train_model(model, X, y)

    save_model(model)