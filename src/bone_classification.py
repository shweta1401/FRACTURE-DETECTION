"""
Bone Classification Training Script (MURA Dataset)
- Supports EfficientNet-B3, EfficientNet-B0, and ResNet-50
- Patient-level stratified split to prevent data leakage
- Includes L1 regularization option
- Outputs metrics, confusion matrices, ROC, PR curves, t-SNE
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, efficientnet_b3, resnet50

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    log_loss, precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE


# -------------------------
# Dataset Class
# -------------------------
class BoneDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------
# Training + Evaluation
# -------------------------
def train_and_evaluate(
    csv_path,
    model_name="efficientnet_b3",
    output_dir="./outputs",
    epochs=15,
    l1_lambda=0.0
):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)
    df["label"] = df["bone_type"].astype("category").cat.codes
    class_names = sorted(df["bone_type"].unique())
    num_classes = len(class_names)

    # Patient-level stratified split
    groups = df["patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=groups, y=df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["patient_id"], y=temp_df["label"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Compute class weights
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(df["label"]), y=df["label"]
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    # Transform (size depends on model)
    img_size = 300 if model_name == "efficientnet_b3" else 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BoneDataset(train_df, transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(BoneDataset(val_df, transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(BoneDataset(test_df, transform), batch_size=32, shuffle=False)

    # -------------------------
    # Model Selection
    # -------------------------
    if model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model_name")

    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    # -------------------------
    # Training Loop
    # -------------------------
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"{model_name} Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{model_name}_classifier.pth"))
            print("💾 Best model saved")

    # -------------------------
    # Evaluation on Test Set
    # -------------------------
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Cohen's Kappa:", cohen_kappa_score(all_labels, all_preds))
    print("Matthews CorrCoef:", matthews_corrcoef(all_labels, all_preds))
    print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))
    print("Log Loss:", log_loss(all_labels, all_probs))
    print("ROC-AUC (Macro):", roc_auc_score(label_binarize(all_labels, classes=range(num_classes)), all_probs, average="macro"))

    # Confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Confusion Matrix (Raw)")
    norm_cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(norm_cm, annot=True, fmt=".2f", ax=axes[1], cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

    # ROC curves
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("Multiclass ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"roc_auc_{model_name}.png"))
    plt.close()

    # t-SNE visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_probs)
    tsne_df = pd.DataFrame(embeddings, columns=["x", "y"])
    tsne_df["label"] = [class_names[i] for i in all_labels]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tsne_df, x="x", y="y", hue="label", palette="tab10")
    plt.title("t-SNE of Output Feature Vectors")
    plt.savefig(os.path.join(output_dir, f"tsne_{model_name}.png"))
    plt.close()


if __name__ == "__main__":
    csv_path = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_all_images.csv"
    output_dir = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\outputs"

    # Example run
    train_and_evaluate(csv_path, model_name="efficientnet_b3", output_dir=output_dir, epochs=15, l1_lambda=1e-5)
