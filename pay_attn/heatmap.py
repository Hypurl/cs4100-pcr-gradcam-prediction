"""
Visualize HiResCAM attention heatmaps overlaid on MRI slices.

Usage (from pay_attn/ or after adding root to sys.path):
    from pay_attn.heatmap import visualize
    visualize(heatmap, img, label, sample_index=0)

heatmap : np.ndarray  shape (1, D', H', W')  — raw output from HiResCam.return_explanation()
img     : torch.Tensor shape (3, 32, 256, 256) — original volume from dataset
label   : int or scalar tensor
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _upsample(heatmap_np: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    ReLU-clamp, normalize to [0,1], then trilinear-upsample to target_size (D, H, W).

    heatmap_np : (1, D', H', W')
    returns    : (D, H, W)
    """
    t = torch.from_numpy(heatmap_np).float()   # (1, D', H', W')
    t = F.relu(t)
    if t.max() > 0:
        t = t / t.max()
    # interpolate expects (N, C, D, H, W)
    t = F.interpolate(
        t.unsqueeze(0),          # (1, 1, D', H', W')
        size=target_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze()                  # (D, H, W)
    return t.numpy()


def _blend(mri_slice: np.ndarray, heat_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a [0,1] grayscale MRI slice with a jet-colormap heatmap.
    Returns an RGB array in [0,1].
    """
    mri_rgb  = np.stack([mri_slice] * 3, axis=-1)
    heat_rgb = cm.jet(heat_slice)[..., :3]
    return (1.0 - alpha) * mri_rgb + alpha * heat_rgb


def visualize(
    heatmap: np.ndarray,
    img: torch.Tensor,
    label,
    sample_index: int = 0,
    pred_prob: float = None,
    time_point: int = 1,
    slices: list = None,
    save_path: str = None,
    show: bool = True,
):
    """
    Render a grid of MRI slices (left) and HiResCAM overlays (right).

    Parameters
    ----------
    heatmap      : np.ndarray, shape (1, D', H', W')
    img          : torch.Tensor, shape (3, 32, 256, 256)
    label        : scalar — ground-truth label (0 or 1)
    sample_index : int — used in the figure title / filename
    pred_prob    : float or None — sigmoid output from the model (0–1); shown in title when provided
    time_point   : int — which of the 3 DCE time channels to show (default 1 = early post-contrast)
    slices       : list[int] — depth indices to display; defaults to 5 evenly spaced slices
    save_path    : str or None — if given, save figure there
    show         : bool — call plt.show()
    """
    img_np = img.numpy() if isinstance(img, torch.Tensor) else img  # (3, 32, 256, 256)
    D = img_np.shape[1]  # 32

    if slices is None:
        slices = [int(round(i)) for i in np.linspace(0, D - 1, 5)]

    heatmap_vol = _upsample(heatmap, target_size=(D, img_np.shape[2], img_np.shape[3]))

    label_val = int(label.item()) if hasattr(label, "item") else int(label)
    actual_str = "PCR" if label_val == 1 else "No PCR"

    # Build prediction line if prob was supplied
    if pred_prob is not None:
        pred_val  = 1 if pred_prob >= 0.5 else 0
        pred_str  = "PCR" if pred_val == 1 else "No PCR"
        correct   = pred_val == label_val
        verdict   = "CORRECT" if correct else "WRONG"
        verdict_color = "green" if correct else "red"
        pred_line = f"Predicted: {pred_str}  ({pred_prob:.3f})  [{verdict}]"
    else:
        verdict_color = "black"
        pred_line = None

    n = len(slices)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.8))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"HiResCAM  |  Sample {sample_index}\n"
        f"Actual: {actual_str}",
        fontsize=13,
        fontweight="bold",
    )

    if pred_line:
        fig.text(0.5, 0.94, pred_line, ha="center", fontsize=10,
                 color=verdict_color, fontweight="bold")

    time_names = ["Pre-contrast", "Early post-contrast", "Late post-contrast"]
    fig.text(0.5, 0.91, f"Time point: {time_names[time_point]}", ha="center", fontsize=9, color="gray")

    for row, depth in enumerate(slices):
        mri_slice  = img_np[time_point, depth]              # (256, 256)
        heat_slice = heatmap_vol[depth]                     # (256, 256)

        # Normalize MRI slice to [0,1] for display
        lo, hi = mri_slice.min(), mri_slice.max()
        mri_norm = (mri_slice - lo) / (hi - lo + 1e-8)

        blended = _blend(mri_norm, heat_slice)

        ax_mri = axes[row][0]
        ax_mri.imshow(mri_norm, cmap="gray", vmin=0, vmax=1)
        ax_mri.set_ylabel(f"Slice {depth}", fontsize=9, rotation=0, labelpad=40, va="center")
        ax_mri.set_xticks([]); ax_mri.set_yticks([])
        if row == 0:
            ax_mri.set_title("Original", fontsize=10)

        ax_cam = axes[row][1]
        ax_cam.imshow(blended, vmin=0, vmax=1)
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        if row == 0:
            ax_cam.set_title("HiResCAM overlay", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    return fig