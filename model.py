
# ───────────────────────────────── Imports ───────────────────────────────────
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (AutoImageProcessor,
                          DeiTForImageClassification
                          )
import matplotlib.pyplot as plt


final_df = pd.read_csv('training_data/full_image_identity.csv')

# ───────────────────────────── Hyper‑parameters ─────────────────────────────
SEED = 42
numEpochs = 50
patience = 3
batchSize = 64
learningRate = 1e-4
weightDecay = 1e-4
freeze_backbone = True
threshold = .6

# ───────────────────────────── Reproducibility ──────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─────────────────────── Prepare dataframe & label encoder ──────────────────
assert 'final_df' in globals(), "final_df is not defined in the notebook scope"
X = final_df['image_path'].values
le = LabelEncoder()
y = le.fit_transform(final_df['identity'].values)
num_classes = len(le.classes_)

# ───────────────────────── Train / val / test split ─────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
X_train, X_val,  y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1875, random_state=SEED)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# ───────────────────────────── Data‑set object ──────────────────────────────


class AnimalDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs


# ─────────────────────────── Model & Optimiser ──────────────────────────────
image_processor = AutoImageProcessor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224", use_fast=True)
model = DeiTForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224",
    num_labels=num_classes,
    id2label={i: lbl for i, lbl in enumerate(le.classes_)},
    label2id={lbl: i for i, lbl in enumerate(le.classes_)},
)

if freeze_backbone:
    for p in model.deit.parameters():
        p.requires_grad = False

model.to(device)
optimizer = AdamW(model.parameters(), lr=learningRate,
                  weight_decay=weightDecay)

# ───────────────────────────── Dataloaders ──────────────────────────────────
train_ds = AnimalDataset(X_train, y_train, image_processor)
val_ds = AnimalDataset(X_val,   y_val,   image_processor)

train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
val_dl = DataLoader(val_ds,   batch_size=batchSize)

test_ds = AnimalDataset(X_test, y_test, image_processor)
test_dl = DataLoader(test_ds, batch_size=batchSize)

# ───────────────────────────── Training loop ───────────────────────────────
best_val_acc = 0.0
no_improve = 0

train_losses = []
val_losses = []
for epoch in range(1, numEpochs + 1):
    # ── Training ──
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    train_loss = running_loss / len(train_dl)
    train_losses.append(train_loss)
    # ── Validation ──

    model.eval()
    val_loss = correct = total = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += preds.size(0)
    val_loss /= len(val_dl)
    val_losses.append(val_loss)
    val_acc = correct / total

    print(
        f"Epoch {epoch:02d} │ train‑loss {train_loss:.4f} │ val‑loss {val_loss:.4f} │ val‑acc {val_acc:.4f}")

    # ── Early stop / checkpoint ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✓ saved new best model")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping validation not improving")
            break

epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.savefig(f"loss_curve.png")

# ───────────────────────────── Testing phase ───────────────────────────────
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch in tqdm(test_dl, desc="Testing"):
        labels = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=1)
        max_probs, pred_idxs = probs.max(dim=1)

        for true_idx, pred_idx, prob in zip(labels, pred_idxs.cpu(), max_probs.cpu()):
            y_true.append(le.classes_[true_idx])
            if prob.item() < threshold:
                y_pred.append("new_identity")
            else:
                y_pred.append(le.classes_[pred_idx.item()])


print("\nTest Classification Report:")
print(classification_report(y_true, y_pred))


labels = le.classes_                   # your class names in order
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 2. Plot the raw counts
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 12))      # adjust size as needed
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
ax.set_title("Confusion Matrix (counts)")
plt.tight_layout()
plt.show()

plt.savefig("ConfusionMatrix.png")
