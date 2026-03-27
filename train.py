"""
Train a CNN using dataset.py
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy    
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
    def __init__(self, learning_rate=LEARNING_RATE, pos_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")
        
    def forward(self, x):
        return self.classifier(self.encoder(x))

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self.shared_step(batch)
        
        self.train_auroc.update(probs, labels.int())
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self.shared_step(batch)
        
        self.val_auroc.update(probs, labels.int())
        self.val_acc.update(probs, labels.int())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def shared_step(self, batch):
        imgs, labels = batch
        logits = self(imgs).squeeze(1)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        return loss, probs, labels
    
    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        
        self.train_auroc.reset()
        
    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        
        self.val_auroc.reset()
        self.val_acc.reset()
    
    def test_step(self, batch, batch_idx):
        loss, probs, labels = self.shared_step(batch)
        
        self.val_auroc.update(probs, labels.int())
        self.val_acc.update(probs, labels.int())
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("test_acc", self.val_acc.compute(), prog_bar=True)
        
        self.val_auroc.reset()
        self.val_acc.reset()
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def main():
    pl.seed_everything(SEED)
    
    training_dataset = BreastDCEDataset(csv_dir=CSVPATH, img_dir=IMGPATH, training_set=True)
    validation_dataset = BreastDCEDataset(csv_dir=CSVPATH, img_dir=IMGPATH, training_set=False)
    
    print(f"Training set length: {len(training_dataset)}")
    print(f"Validation set length: {len(validation_dataset)}")
    
    labels = training_dataset.metadata["pCR"].values
    
    num_pos = labels.sum()
    num_neg = len(labels) - num_pos
    pos_weight = num_neg / num_pos
    
    print(f"There exist {num_pos} positive samples and {num_neg} negative samples")
    print(f"pos_weight = {pos_weight}")
    
    # Change num_workers and pin_memory if running on windows with GPU
    training_dataloader = DataLoader(
        training_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    ) 
    
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )
    
    model = pcrCNN(learning_rate=LEARNING_RATE, pos_weight=pos_weight)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=8,
        mode="min",
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1
    )
    
    trainer.fit(model, training_dataloader, validation_dataloader)
    
    torch.save(pcrCNN.load_from_checkpoint(checkpoint_callback.best_model_path, weights_only=False).state_dict(), "model.pth")
    

if __name__ == "__main__":
    main()