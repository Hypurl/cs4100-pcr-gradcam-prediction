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

PATHS = [
    ("BreastDCEDL_ISPY1_min_crop", "dce"),
    ("BreastDCEDL_ISPY2_min_crop", "dce"),
    ("BreastDCEDL_DUKE_min_crop",  "crop_min_dce")
]

def get_path(pid, data_dir):
    for ds, dce in PATHS:
        matches = glob.glob(os.path.join(data_dir, ds, dce, f"{pid}*.nii.gz"))
        
        if matches:
            return matches[0]

class BreastDCEDataset(Dataset):
    def __init__(self, csv_dir, data_dir, training_set=True):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(csv_dir)
        
        self.metadata['pid'] = self.metadata['pid'].astype(str)
        
        # drops all entries missing pcr or test
        before = len(self.metadata)
        self.metadata = self.metadata.dropna(subset=["pCR", "test"])
        
        if before - len(self.metadata):
            print(f"{before - len(self.metadata)} entries dropped for missing pCR or test")
        
        if training_set:
            self.metadata = self.metadata[self.metadata['test'].astype(int) == 0]
        else:
            self.metadata = self.metadata[self.metadata['test'].astype(int) == 1]
            
        self.metadata = self.metadata.reset_index(drop=True)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        pid = self.metadata.loc[index, 'pid']
        label = self.metadata.loc[index, 'pCR']
        
        path = get_path(pid, self.data_dir)
        nib_img = nibabel.load(path).get_fdata().astype(np.float32)
        
        nib_img = np.transpose(nib_img, (2, 1, 0))
        nib_img = np.expand_dims(nib_img, axis=0) 
        
        img_tensor = torch.from_numpy(nib_img)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(256, 256, 32), mode='trilinear', align_corners=False).squeeze(0)
        
        # normalize voxel to [0, 1]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        
        return img_tensor, label_tensor

# main for testing
if __name__ == "__main__":
    CSVPATH = "./data/BreastDCEDL_metadata_min_crop.csv"
    DATAPATH = "./data"
    
    dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, training_set=True)

    print(f"{len(dataset)} MRI scans found.")
    
    # Show first mri photo
    img = dataset[0][0].squeeze(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    for i, s in enumerate([0, 15, 31]):
        axes[i].imshow(img[:, :, s], cmap="gray")
        axes[i].set_title(f"Slice {s+1}")
        axes[i].axis("off")
        
    plt.show()