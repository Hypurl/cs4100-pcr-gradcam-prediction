"""
Evaluate the model across the full test set and report AUROC, accuracy,
and the distribution of predicted probabilities.

    cd pay_attn && python eval.py
"""

import os, sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from train import PcrCNN
from dataset import BreastDCEDataset, Split

CSV_PATH   = os.path.join(_root, "data/BreastDCEDL_metadata_min_crop.csv")
DATA_PATH  = os.path.join(_root, "data")
MODEL_PATH = os.path.join(_root, "model_samples/model_best_auroc.pth")

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PcrCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    from dataset import get_path
    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        ds = BreastDCEDataset(csv_dir=CSV_PATH, data_dir=DATA_PATH, split=split)
        n_found = sum(1 for i in range(len(ds)) if get_path(ds.metadata.loc[i, 'pid'], DATA_PATH))
        print(f"{split.name:5s}: {len(ds):3d} in CSV,  {n_found:3d} files on disk")
    print()

    dataset = BreastDCEDataset(csv_dir=CSV_PATH, data_dir=DATA_PATH, split=Split.TEST)
    print(f"Test set size: {len(dataset)}")

    probs, labels = [], []
    skipped = 0

    for i in range(len(dataset)):
        try:
            img, label = dataset[i]
        except TypeError:
            skipped += 1
            continue

        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(device)).squeeze()
            prob  = torch.sigmoid(logit).item()

        probs.append(prob)
        labels.append(int(label.item()))

    print(f"Skipped (missing files): {skipped}")
    print(f"Evaluated: {len(probs)} samples\n")

    probs_np  = np.array(probs)
    labels_np = np.array(labels)
    preds_np  = (probs_np >= 0.5).astype(int)

    auroc = roc_auc_score(labels_np, probs_np)
    acc   = accuracy_score(labels_np, preds_np)
    cm    = confusion_matrix(labels_np, preds_np)
    pcr_rate = labels_np.mean()

    print(f"PCR rate (base rate) : {pcr_rate:.3f}")
    print(f"AUROC                : {auroc:.4f}")
    print(f"Accuracy             : {acc:.4f}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred No PCR  Pred PCR")
    print(f"Actual No PCR     {cm[0,0]:<6}       {cm[0,1]}")
    print(f"Actual PCR        {cm[1,0]:<6}       {cm[1,1]}")

    print(f"\nPredicted prob stats:")
    print(f"  mean  : {probs_np.mean():.4f}")
    print(f"  std   : {probs_np.std():.4f}")
    print(f"  min   : {probs_np.min():.4f}")
    print(f"  max   : {probs_np.max():.4f}")

    # ── plot prediction distribution ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Test-set evaluation  |  AUROC: {auroc:.3f}  |  Acc: {acc:.3f}", fontsize=13)

    # Histogram: predicted prob split by true label
    ax = axes[0]
    ax.hist(probs_np[labels_np == 0], bins=20, alpha=0.6, label="Actual: No PCR", color="steelblue")
    ax.hist(probs_np[labels_np == 1], bins=20, alpha=0.6, label="Actual: PCR",    color="tomato")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Predicted PCR probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction distribution by true label")
    ax.legend()

    # Confusion matrix heatmap
    ax2 = axes[1]
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred No PCR", "Pred PCR"])
    ax2.set_yticklabels(["Actual No PCR", "Actual PCR"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.set_title("Confusion matrix")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    save_path = os.path.join(_root, "eval_test_set.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()