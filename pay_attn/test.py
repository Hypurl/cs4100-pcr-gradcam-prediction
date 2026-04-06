import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
import torch
from train import PcrCNN
from hirescam import HiResCam
import numpy as np
from dataset import BreastDCEDataset, Split

dataset = BreastDCEDataset(csv_dir=os.path.join(_root, "data/BreastDCEDL_metadata_min_crop.csv"), data_dir=os.path.join(_root, "data"), split=Split.TEST)

# 1. Load the trained model
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = PcrCNN()
model.load_state_dict(torch.load(os.path.join(_root, 'model_samples/model_best_auroc.pth'), map_location=device))
model.to(device)

# 2. Set up HiResCam
#    target_layer_name '3' = last ConvBlock in encoder
cam = HiResCam(
    model=model,
    device=device,
    model_name='PcrCNN',
    target_layer_name='3'
)

# 3. Get a sample input (shape: 1, 3, 32, 256, 256)
#    Load from your dataset however you normally do it
img, label = dataset[0]
ctvol = img.unsqueeze(0).to(device)
# 4. Run HiResCam
heatmap = cam.return_explanation(
    ctvol=ctvol,
    chosen_label_index=0 # only one class
)

# heatmap shape: (1, 2, 16, 16) — spatial map over the volume
print("Heatmap shape:", heatmap.shape)
print("Heatmap min:", heatmap.min())
print("Heatmap max:", heatmap.max())
print("Any NaN:", np.isnan(heatmap).any())
print("All zeros:", (heatmap == 0).all())