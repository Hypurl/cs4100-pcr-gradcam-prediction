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

class BreastDCEDataset(Dataset):
    def __init__(self, csv_dir, img_dir, training_set = True):
        self.img_dir = img_dir
        self.metadata = pd.read_csv(csv_dir)
        
        # TODO: Update to use more than ISPY1
        self.metadata['pid'] = self.metadata['pid'].astype(str)
        self.metadata = self.metadata[self.metadata['pid'].str.startswith('ISPY1')]
        
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
        
        matches = glob.glob(os.path.join(self.img_dir, 'dce', f"{pid}*.nii.gz"))
        
        nib_img = nibabel.load(matches[0]).get_fdata().astype(np.float32)
        
        nib_img = np.transpose(nib_img, (2, 1, 0))
        nib_img = np.expand_dims(nib_img, axis=0) 
        
        img_tensor = torch.from_numpy(nib_img)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256, 32), mode='trilinear', align_corners=False)
        
        # TODO: Normalize voxel values
        
        return img_tensor, label_tensor
    