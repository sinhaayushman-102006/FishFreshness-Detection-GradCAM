import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

# ========================
# CONFIG
# ========================
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# LOAD PRETRAINED WEIGHTS
# ========================
weights = EfficientNet_V2_S_Weights.DEFAULT

# ========================
# TRANSFORMS (IMPORTANT)
# ========================
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    weights.transforms()
])

val_transforms = weights.transforms()

# ========================
# DATASETS
# ========================
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_names = train_dataset.classes
print("Classes:", class_names)

# ========================
# MODEL
# ========================
model = efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 3)

model = model.to(DEVICE)

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ========================
# TRAIN FUNCTION
# ========================
def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(train_loader), acc

# ========================
# VALIDATION FUNCTION
# ========================
def validate():
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    return acc, all_preds, all_labels

# ========================
# TRAIN LOOP
# ========================
best_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_acc, preds, labels = validate()

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

# ========================
# FINAL METRICS
# ========================
print("\n📊 FINAL REPORT")
print(classification_report(labels, preds, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds)) 