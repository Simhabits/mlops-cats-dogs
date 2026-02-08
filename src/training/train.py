import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Config
# -----------------------
DATA_DIR = "datasets/cats_dogs"
BATCH_SIZE = 32
EPOCHS = 3
LR = 0.001
MODEL_DIR = "output"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Data Pipeline
# -----------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes

# -----------------------
# Model Definition
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 31 * 31, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# MLflow Setup
# -----------------------
mlflow.set_experiment("Cats_vs_Dogs_Baseline")

all_preds = []
all_labels = []

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LR)

    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        epoch_loss = 0.0

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        accuracy = correct / total
        avg_loss = epoch_loss / len(dataloader)

        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("loss", avg_loss, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    # -----------------------
    # Save Model
    # -----------------------
    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.pytorch.log_model(model, "model")

    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

print("Training complete. Model saved.")
