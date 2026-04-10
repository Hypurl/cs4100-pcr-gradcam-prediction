"""
dataset.py - Breast DCE-MRI Data Loading
=========================================

Loads DCE breast MRI volumes from ISPY-1, ISPY-2, DUKE, and 
resamples them to same resolutions and outputs (image, label)
pairs for pCR prediction.

Download the min crop dataset from https://zenodo.org/records/18114231. 

.. warning:: DO NOT COMMIT DATASET.
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
"""list[tuple[str, str]]: Lookup table matching source dataset to directory path. Each 
dataset has their own unique subdirectory names."""

class Split(IntEnum):
    """Integer enum representing data splits.
    
    The metadata csv contains test column with values 0, 1, or 2, corresponding to
    training, validation, and test partitions.
    
    .. attribute:: TRAIN
       :value: 0
 
       Training split.
 
    .. attribute:: VAL
       :value: 1
 
       Validation split.
 
    .. attribute:: TEST
       :value: 2
 
        Test split.
    """
    TRAIN = 0
    VAL = 1
    TEST = 2

def get_path(pid, data_dir):
    """Get path of the three DCE NIfTI files for a given patient.
    
    :param pid: Patient ID used to match filenames.
    :type pid: str
    :param data_dir: Root directory of the dataset folders.
    :type data_dir: str
    :returns: A list of three file paths if found, else None if no dataset
        directory contains at least three matching files.
    :rtype: list[str] | None
    
    .. note:: If a patient has more than three-point files, only the first three,
        sorted alphabetically, are used."""
    for ds, dce in PATHS:
        matches = sorted(glob.glob(os.path.join(data_dir, ds, dce, f"{pid}*.nii.gz")))
        if len(matches) >= 3:
            return matches[:3]
    
    return None

class BreastDCEDataset(Dataset):
    """PyTorch :class:`~torch.utils.data.Dataset` of preprocessed DCE-MRI
    volumes and labels.
    
    Each item is a tuple (image_tensor, label_tensor) where the image has 
    shape (3, 32, 256, 256) with three contrast-phase channels each sampled
    to 32 slices of 256x256 pixels and the label is a float (1.0 for pCR, 0.0 otherwise). 
    
    See main for example usage.
    """
    
    def __init__(self, csv_dir, data_dir, split=Split.TRAIN):
        """
        :param csv_dir: Path to metadata CSV file. It should at least contain
            columns pid, pCR, and test.
        :type csv_dir: str
        :param data_dir: Root directory of the dataset folders listed in :data:`PATHS`.
        :type data_dir: str
        :param split: Which data split to load.
        :type split: :class:`Split`
        """
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
        """Return the number of valid samples in split.
        
        :returns: Sample count.
        :rtype: int
        """
        return len(self.metadata)
    
    def __getitem__(self, index):
        """Load and return a single sample by:
        
        - Finding patient ID and pCR label from :attr:`metadata`.
        - Load three NIfTI time point volumes by :func:`get_path` and
            converting each to float32
        - Stack into shape of (3, D, H, W) where the channels are the
            three contrast phases: pre, early-post, late-post.
        - Resize to (3, 32, 256, 256) by trilinear interpolation.
        - Min-max normalizes all three channels.
        
        :param index: Row index of :attr:`metadata`.
        :type index: int
        :returns: A tuple of (image_tensor, label_tensor).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
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