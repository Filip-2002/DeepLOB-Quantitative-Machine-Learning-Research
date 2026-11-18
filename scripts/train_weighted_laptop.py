from pathlib import Path
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, balanced_accuracy_score
from scripts.train_extension import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import train_one_epoch, save_checkpoint

config = yaml.safe_load(
    Path("configs/deeplob_laptop_diverse.yaml").read_text()
)

torch.manual_seed(42)
device = torch.device("cpu")

train_loader, val_loader, _ = get_dataloaders(
    data_dir=config["data"]["data_dir"],
    horizon=10,
    batch_size=64
)

# Moderate inverse-square-root class weighting
counts = torch.bincount(
    train_loader.dataset.y, minlength=3
).float()
weights = counts.clamp_min(1).rsqrt()
weights /= weights.mean()

print("Class weights:", weights.tolist())

model = build_model(config).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, eps=1e-8
)

out = Path("experiments/laptop_weighted/k10")
out.mkdir(parents=True, exist_ok=True)

best_f1 = -1
bad_epochs = 0

for epoch in range(1, 17):
    loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    model.eval()
    predictions = []
    labels = []

    with torch.inference_mode():
        for x, y in val_loader:
            logits = model(x.to(device))
            predictions.append(logits.argmax(1).cpu())
            labels.append(y.cpu())

    pred = torch.cat(predictions).numpy()
    actual = torch.cat(labels).numpy()

    acc = (pred == actual).mean()
    macro_f1 = f1_score(actual, pred, average="macro")
    balanced = balanced_accuracy_score(actual, pred)

    print(
        f"Epoch {epoch}/16 | Train: {train_acc:.4f} | "
        f"Val acc: {acc:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Balanced: {balanced:.4f}", flush=True
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        bad_epochs = 0
        save_checkpoint(
            model, optimizer, epoch, float(acc),
            out / "best_model.pt"
        )
        print("  New best checkpoint saved")
    else:
        bad_epochs += 1

    if bad_epochs >= 5:
        print("Early stopping")
        break

print(f"Best validation macro-F1: {best_f1:.4f}")
