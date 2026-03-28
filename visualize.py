"""
Visualize HiResCAM attention maps overlaid on MRI slices for a single patient.

Pipeline:
    1. Load model + one test sample
    2. Run HiResCAM (via hirescam.py) at the last conv layer of the encoder
    3. Upsample the 3D heatmap to the input volume size
    4. For each of the 3 time points, show MRI slices side-by-side with the heatmap overlay

Note: The model ends in two FC layers, so HiResCAM's faithfulness guarantee
(paper Section 3.5) does not strictly apply, but it is still more faithful than Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import BreastDCEDataset, Split
from train import pcrCNN
from hirescam import HiResCam  #not implemented yet

CSVPATH   = "./data/BreastDCEDL_metadata_min_crop.csv"
DATAPATH  = "./data"
MODEL_PATH = "./model_samples/model_best_auroc.pth"

SAMPLE_INDEX = 0          # which test sample to visualize
SLICES       = [0, 15, 31]  # depth indices to show (out of 32)
TIME_NAMES   = ["Pre-contrast", "Early post-contrast", "Late post-contrast"]

#Helper functions
def load_model(path):
    model = pcrCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def upsample_heatmap(heatmap: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    ReLU + normalize to [0,1] + trilinear upsample to target_size (D, H, W).
    heatmap: shape (D', H', W')
    """
    heatmap = F.relu(heatmap)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze()
    return heatmap


def overlay_heatmap_on_slice(mri_slice: np.ndarray, heat_slice: np.ndarray, alpha=0.5):
    """
    Blend a grayscale MRI slice with a jet-colormap heatmap.
    Both inputs should be in [0, 1].
    Returns an RGB array.
    """
    mri_rgb  = np.stack([mri_slice] * 3, axis=-1)
    heat_rgb = cm.jet(heat_slice)[..., :3]
    return (1 - alpha) * mri_rgb + alpha * heat_rgb



#Main
def main():
    #Data handling
    dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, split=Split.TEST)
    img, label = dataset[SAMPLE_INDEX] 
    img_batch = img.unsqueeze(0) 
    print(f"Sample index : {SAMPLE_INDEX}")
    print(f"True label   : {'PCR' if label.item() == 1 else 'No PCR'}")

    #Use sample models if on mac
    model = load_model(MODEL_PATH)


    #PLACEHOLDER: replace with real HiResCam call once hirescam.py is done
    cam_engine = HiResCam(model, target_layer=model.encoder[3])
    raw_cam = cam_engine.compute(img_batch)
   
    #Upsample heatmap to input volume dimensions (32, 256, 256)
    heatmap = upsample_heatmap(raw_cam, target_size=(32, 256, 256))  # (32, 256, 256)
    heatmap_np = heatmap.detach().numpy()

    #Predicted probability
    with torch.no_grad():
        logit = model(img_batch).squeeze()
        prob  = torch.sigmoid(logit).item()
    print(f"Predicted PCR probability: {prob:.4f}")

    #plot visuals
    n_slices = len(SLICES)
    n_timepoints = 3
    fig, axes = plt.subplots(
        n_timepoints * 2, n_slices,
        figsize=(n_slices * 4, n_timepoints * 2 * 3)
    )
    fig.suptitle(
        f"HiResCAM — Sample {SAMPLE_INDEX} | "
        f"True: {'PCR' if label.item()==1 else 'No PCR'} | "
        f"Pred: {prob:.3f}",
        fontsize=14
    )

    img_np = img.numpy() #(3, 32, 256, 256)

    for t, time_name in enumerate(TIME_NAMES):
        for s_idx, depth in enumerate(SLICES):
            mri_slice  = img_np[t, depth] # (256, 256), in [0,1]
            heat_slice = heatmap_np[depth] # (256, 256), in [0,1]

            # Row 2t: raw MRI slice
            ax_mri = axes[t * 2][s_idx]
            ax_mri.imshow(mri_slice, cmap="gray", vmin=0, vmax=1)
            ax_mri.set_title(f"{time_name}\nSlice {depth + 1}", fontsize=9)
            ax_mri.axis("off")

            # Row 2t+1: MRI + heatmap overlay
            ax_cam = axes[t * 2 + 1][s_idx]
            blended = overlay_heatmap_on_slice(mri_slice, heat_slice)
            ax_cam.imshow(blended)
            ax_cam.set_title("HiResCAM overlay", fontsize=9)
            ax_cam.axis("off")

    plt.tight_layout()
    plt.savefig(f"hirescam_sample_{SAMPLE_INDEX}.png", dpi=150)
    plt.show()
    print(f"Saved to hirescam_sample_{SAMPLE_INDEX}.png")


if __name__ == "__main__":
    main()
