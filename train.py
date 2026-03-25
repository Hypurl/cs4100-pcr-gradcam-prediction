"""
Traub a CNN using dataset.py
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import BreastDCEDataset

CSVPATH = "./data/BreastDCEDL_metadata_min_crop.csv"
IMGPATH = "./data/BreastDCEDL_ISPY1_min_crop"

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 20

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