"""
Fracture Detection Training Script (MURA Dataset)
- Trains binary classifiers (fracture vs no fracture) per bone type
- Supports ResNet-50, EfficientNet-B0, EfficientNet-B3
- Uses stratified train/val split
- Saves best models per bone type
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, efficientnet_b3, resnet50
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


# -------------------------
# Dataset Class
# -------------------------
class FractureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = int(self.df.iloc[idx]["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------
# Training for One Bone Type
# -------------------------
def run_model(df_bone, arch="resnet50", bone_name="XR_ELBOW",
              img_size=224, batch_size=32, epochs=10, lr=1e-4,
              output_dir="./fracture_models"):
    """
    Train and evaluate model for one bone type.
    Args:
        df_bone (pd.DataFrame): Subset dataframe for a single bone type
        arch (str): Model architecture ('resnet50', 'efficientnet_b0', 'efficientnet_b3')
        bone_name (str): Bone type string
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train/Val Split
    train_df, val_df = train_test_split(
        df_bone, test_size=0.2, stratify=df_bone["label"], random_state=42
    )

    # Dataset + DataLoader
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = FractureDataset(train_df, transform)
    val_ds = FractureDataset(val_df, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    # -------------------------
    # Model Selection
    # -------------------------
    if arch == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif arch == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        img_size = 300  # EfficientNet-B3 default input size
    else:
        raise ValueError("Unsupported architecture")

    model = model.to(device)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_correct, train_loss = 0, 0.0
        for images, labels in tqdm(train_loader, desc=f"[{arch}-{bone_name}] Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_loss += loss.item() * images.size(0)

        train_acc = train_correct / len(train_loader.dataset)
        print(f"✅ Train Accuracy: {train_acc:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"🧪 Val Accuracy: {val_acc:.4f} | F1 Score: {val_f1:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(output_dir, f"{arch}_{bone_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved Best Model: {model_path}")

    # Final Classification Report
    print("\n--- Final Evaluation ---")
    print(classification_report(val_labels, val_preds, target_names=["No Fracture", "Fracture"]))


# -------------------------
# Train Across All Bones
# -------------------------
def train_all(csv_path, architectures=None, output_dir="./fracture_models"):
    """
    Train fracture detection models for all bones using given architectures.
    Args:
        csv_path (str): Path to CSV with image_path, bone_type, label
        architectures (list): Architectures to train (default: all three)
    """
    if architectures is None:
        architectures = ["resnet50", "efficientnet_b0", "efficientnet_b3"]

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"fracture_label": "label"})
    df["label"] = df["label"].astype(int)

    bone_types = df["bone_type"].unique()
    for bone in bone_types:
        df_bone = df[df["bone_type"] == bone].reset_index(drop=True)
        for arch in architectures:
            print(f"\n🔥 Training {arch} for {bone}")
            run_model(df_bone, arch=arch, bone_name=bone, output_dir=output_dir)


if __name__ == "__main__":
    csv_path = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\labels_combined.csv"
    output_dir = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\fracture_models"

    train_all(csv_path, architectures=["resnet50", "efficientnet_b0", "efficientnet_b3"], output_dir=output_dir)
