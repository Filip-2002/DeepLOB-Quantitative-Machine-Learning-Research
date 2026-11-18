from pathlib import Path
import numpy as np
import torch

root = Path.cwd()
raw = root / "data" / "raw"
out = root / "data" / "processed_diverse"
out.mkdir(exist_ok=True)

def make_tensors(data, indices):
    features = data[:40].T
    labels = data[-5:].T.astype(np.int64) - 1

    X = torch.empty((len(indices), 100, 40), dtype=torch.float32)
    Y = torch.empty((len(indices), 5), dtype=torch.int64)

    for j, end in enumerate(indices):
        X[j] = torch.from_numpy(
            np.ascontiguousarray(features[end-99:end+1])
        )
        Y[j] = torch.from_numpy(labels[end].copy())

    return X, Y

def save_split(name, X, Y):
    torch.save(X, out / f"{name}_x.pt")
    torch.save(Y, out / f"{name}_y.pt")
    print(f"{name}: {len(X):,} samples")

print("Loading all seven training days...")
data = np.loadtxt(
    raw / "Train_Dst_NoAuction_DecPre_CF_7.txt",
    dtype=np.float32
)

n = data.shape[1]
split = int(n * 0.8)

# Sample every 8th window throughout the period.
# Reserve a gap around the chronological split.
train_idx = np.arange(99, split - 100, 8)
val_idx = np.arange(split + 100, n - 100, 8)

X, Y = make_tensors(data, train_idx)
save_split("train", X, Y)
del X, Y

X, Y = make_tensors(data, val_idx)
save_split("val", X, Y)
del X, Y, data

# Keep separate test days separate when forming windows.
# Use the first 5,000 observations from each test file.
test_X, test_Y = [], []

for day in (7, 8, 9):
    data = np.loadtxt(
        raw / f"Test_Dst_NoAuction_DecPre_CF_{day}.txt",
        dtype=np.float32
    )[:, :5000]

    indices = np.arange(99, data.shape[1] - 100)
    X, Y = make_tensors(data, indices)
    test_X.append(X)
    test_Y.append(Y)
    del data

save_split("test", torch.cat(test_X), torch.cat(test_Y))

print("Diverse dataset created successfully.")
