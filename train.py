"""
Train a CNN using dataset.py
"""

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BreastDCEDataset

CSVPATH = "./data/BreastDCEDL_metadata_min_crop.csv"
IMGPATH = "./data/BreastDCEDL_ISPY1_min_crop"

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 20
SEED = 67

class ConvBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(num_input_channels, num_output_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(num_output_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2)
                                   )
        
        def forward(self, x):
            return self.block(x)

class pcrCNN(pl.LightningModule):
    
    # TODO
    def __init__(self):
        super().__init__()
        
    # TODO: forward passes
    def forward(self, x):
        pass
    
    # TODO: find training loss
    def training_step(self, batch, batch_idx):
        pass
    
    # TODO: optimizer algorithm
    def configure_optimizers(self):
        pass

def main():
    # TODO 
    pass

if __name__ == "__main__":
    main()