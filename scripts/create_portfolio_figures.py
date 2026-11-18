from pathlib import Path
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_fscore_support
)
from src.models.deeplob import DeepLOB

folder = Path("experiments/laptop_stride4/k10")
out = Path("experiments/laptop_stride4/figures")
out.mkdir(parents=True, exist_ok=True)

history = torch.load(folder / "history.pt", weights_only=True)
checkpoint = torch.load(
    folder / "best_model.pt",
    map_location="cpu", weights_only=True
)
config = yaml.safe_load((folder / "config.yaml").read_text())
m = config["model"]

# Figure 1: Training and validation accuracy
epochs = np.arange(1, len(history["train_accuracy"]) + 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, np.array(history["train_accuracy"]) * 100,
        label="Training", linewidth=2)
ax.plot(epochs, np.array(history["val_accuracy"]) * 100,
        label="Validation", linewidth=2)
ax.axvline(checkpoint["epoch"], linestyle="--",
           label=f'Best checkpoint: epoch {checkpoint["epoch"]}')
ax.set(xlabel="Epoch", ylabel="Accuracy (%)",
       title="DeepLOB | Training and Validation Accuracy")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "01_learning_curves.png", dpi=220)
plt.close(fig)

# Load saved model
model = DeepLOB(
    num_classes=m["num_classes"],
    conv_filters=m["conv_filters"],
    inception_filters=m["inception_filters"],
    lstm_hidden=m["lstm_hidden"],
    lstm_layers=m["lstm_layers"],
    leaky_relu_slope=m["leaky_relu_slope"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

X = torch.load(
    "data/processed_stride4/val_x.pt", weights_only=True
)
Y = torch.load(
    "data/processed_stride4/val_y.pt", weights_only=True
)[:, 0].numpy()

predictions = []
with torch.inference_mode():
    for i in range(0, len(X), 64):
        logits = model(X[i:i+64].unsqueeze(1))
        predictions.extend(logits.argmax(1).tolist())

names = ["Down", "Stationary", "Up"]

# Figure 2: Confusion matrix
cm = confusion_matrix(Y, predictions, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("DeepLOB | Validation Confusion Matrix")
fig.tight_layout()
fig.savefig(out / "02_confusion_matrix.png", dpi=220)
plt.close(fig)

# Figure 3: Class-level metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    Y, predictions, labels=[0, 1, 2], zero_division=0
)

x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x-width, precision*100, width, label="Precision")
ax.bar(x, recall*100, width, label="Recall")
ax.bar(x+width, f1*100, width, label="F1-score")
ax.set_xticks(x, names)
ax.set_ylim(0, 105)
ax.set_ylabel("Score (%)")
ax.set_title("DeepLOB | Performance by Price Movement")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(out / "03_class_metrics.png", dpi=220)
plt.close(fig)

print("All 3 figures saved to:", out.resolve())
print(classification_report(
    Y, predictions, labels=[0,1,2],
    target_names=names, zero_division=0
))
