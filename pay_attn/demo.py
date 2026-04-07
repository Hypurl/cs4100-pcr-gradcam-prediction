"""
Run HiResCAM on a random test-set sample and display the heatmap.

    cd pay_attn && python demo.py
    cd pay_attn && python demo.py --index 5   # specific sample
"""

import os, sys, argparse, random
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from train import PcrCNN
from hirescam import HiResCam
from heatmap import visualize
from dataset import BreastDCEDataset, Split

# ── paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(_root, "data/BreastDCEDL_metadata_min_crop.csv")
DATA_PATH  = os.path.join(_root, "data")
MODEL_PATH = os.path.join(_root, "model_samples/model_best_auroc.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                        help="Test-set index (omit for random)")
    args = parser.parse_args()

    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = BreastDCEDataset(csv_dir=CSV_PATH, data_dir=DATA_PATH, split=Split.TEST)
    print(f"Test set size: {len(dataset)}")

    indices = list(range(len(dataset)))
    if args.index is not None:
        indices = [args.index]
    else:
        random.shuffle(indices)

    img, label, idx = None, None, None
    for candidate in indices:
        try:
            img, label = dataset[candidate]
            idx = candidate
            break
        except TypeError:
            print(f"Sample {candidate} missing data files, skipping...")

    if img is None:
        print("No loadable samples found.")
        return

    print(f"Sample index : {idx}")
    label_str  = "PCR" if label.item() == 1 else "No PCR"
    print(f"True label   : {label_str}")

    # ── model ────────────────────────────────────────────────────────────────
    model = PcrCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # ── predicted probability ────────────────────────────────────────────────
    with torch.no_grad():
        logit = model(img.unsqueeze(0).to(device)).squeeze()
        prob  = torch.sigmoid(logit).item()
    print(f"Pred PCR prob: {prob:.4f}")

    # ── HiResCAM ─────────────────────────────────────────────────────────────
    cam = HiResCam(model=model, device=device, model_name="PcrCNN", target_layer_name="3")
    heatmap = cam.return_explanation(
        ctvol=img.unsqueeze(0).to(device),
        chosen_label_index=0,
    )
    print(f"Heatmap shape: {heatmap.shape}")

    # ── visualize ────────────────────────────────────────────────────────────
    save_path = os.path.join(_root, f"hirescam_sample_{idx}.png")
    visualize(
        heatmap=heatmap,
        img=img,
        label=label,
        sample_index=idx,
        pred_prob=prob,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
