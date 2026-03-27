"""
Download the min crop dataset from https://zenodo.org/records/18114231. 

DO NOT COMMIT DATASET.
"""
import glob
import nibabel
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from enum import IntEnum

PATHS = [
    ("BreastDCEDL_ISPY1_min_crop", "dce"),
    ("BreastDCEDL_ISPY2_min_crop", "dce"),
    ("BreastDCEDL_DUKE_min_crop",  "crop_min_dce")
]

class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2

def get_path(pid, data_dir):
    for ds, dce in PATHS:
        matches = sorted(glob.glob(os.path.join(data_dir, ds, dce, f"{pid}*.nii.gz")))
        if len(matches) >= 3:
            return matches[:3]
    
    return None

class BreastDCEDataset(Dataset):
    def __init__(self, csv_dir, data_dir, split=Split.TRAIN):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(csv_dir)
        
        self.metadata['pid'] = self.metadata['pid'].astype(str)
        
        # drops all entries missing pcr or test
        before = len(self.metadata)
        self.metadata = self.metadata.dropna(subset=["pCR", "test"])
        
        if before - len(self.metadata):
            print(f"{before - len(self.metadata)} entries dropped for missing pCR or test")
        
        self.metadata = self.metadata[self.metadata['test'].astype(int) == split.value]
            
        self.metadata = self.metadata.reset_index(drop=True)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        pid = self.metadata.loc[index, 'pid']
        label = self.metadata.loc[index, 'pCR']
        
        paths = get_path(pid, self.data_dir)

        stack = []
        for p in paths:
            stack.append(nibabel.load(p).get_fdata().astype(np.float32))

        img = np.stack(stack, axis=0)

        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(32, 256, 256),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        # normalize voxel to [0, 1]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        
        return img_tensor, label_tensor

# main for testing
if __name__ == "__main__":
    CSVPATH = "./data/BreastDCEDL_metadata_min_crop.csv"
    DATAPATH = "./data"
    
    dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, split=Split.TRAIN)

    print(f"{len(dataset)} MRI scans found.")
    
    img, label = dataset[0]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    names = ["Pre-contrast", "Early post-contrast", "Late post-contrast"]

    for n in range(3):
        for i, j in enumerate([0, 15, 31]):
            axes[n][i].imshow(img[n, j, :, :].numpy(), cmap="gray")
            axes[n][i].set_title(f"{names[n]}: Slice {j + 1}")
            axes[n][i].axis("off")

    plt.show()