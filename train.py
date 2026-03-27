"""
Train a CNN using dataset.py
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy    
from dataset import BreastDCEDataset, Split  
import numpy as np

# from sklearn.model_selection import train_test_split  

torch.set_float32_matmul_precision("high")

CSVPATH = "./data/BreastDCEDL_metadata_min_crop.csv"
DATAPATH = "./data"

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 30
NUM_WORKERS = 8
PERSISTENT_WORKERS = bool(NUM_WORKERS)
SEED = 67

# VALIDATION_SPLIT = 0.2

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
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")

        self.test_auroc = AUROC(task="binary")
        self.test_acc = Accuracy(task="binary")

        
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
        
        self.test_auroc.update(probs, labels.int())
        self.test_acc.update(probs, labels.int())
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute(), prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        
        self.test_auroc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.001
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"}
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Checks for NaN weights at the end of every training batch."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"\n[CRITICAL WARNING] NaN weights detected in layer: {name} at batch {batch_idx}!")
                # Optional: break to stop spamming the console
                break

def main():
    pl.seed_everything(SEED)
    
    # OLD IMPLEMENTATION
    """
    # Validation takes 20% of training set rather than using the test set
    full_set = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, training_set=True)
    labels = full_set.metadata["pCR"].values.astype(int)
    
    train_index, val_index = train_test_split(
        np.arange(len(full_set)),
        test_size=VALIDATION_SPLIT,
        stratify=labels,
        random_state=SEED
    )
    
    training_dataset = torch.utils.data.Subset(full_set, train_index)
    validation_dataset = torch.utils.data.Subset(full_set, val_index)
    
    print(f"Training set length: {len(training_dataset)}")
    print(f"Validation set length: {len(validation_dataset)}")
    """

    training_dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, split=Split.TRAIN)
    validation_dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, split=Split.VAL)
    test_dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, split=Split.TEST)

    print(f"Training set length: {len(training_dataset)}")
    print(f"Validation set length: {len(validation_dataset)}")
    print(f"Test set length: {len(test_dataset)}")

    labels = training_dataset.metadata["pCR"].values.astype(int)
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS
    ) 
    
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS//2,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    model = pcrCNN(learning_rate=LEARNING_RATE, pos_weight=pos_weight)
    
    ckpt_auroc = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="best-auroc",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    ckpt_loss = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="best-loss",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=8,
        mode="max",
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda",
        precision="32",
        devices=1,
        callbacks=[ckpt_auroc, ckpt_loss, early_stop_callback],
        log_every_n_steps=1,
        detect_anomaly=True
    )
    
    trainer.fit(model, training_dataloader, validation_dataloader)

    best_auroc = pcrCNN.load_from_checkpoint(ckpt_auroc.best_model_path, weights_only=False)
    torch.save(best_auroc.state_dict(), "model_best_auroc.pth")

    best_loss = pcrCNN.load_from_checkpoint(ckpt_loss.best_model_path, weights_only=False)
    torch.save(best_loss.state_dict(), "model_best_loss.pth")
    
    # test_dataset = BreastDCEDataset(csv_dir=CSVPATH, data_dir=DATAPATH, training_set=False)

    best_auroc_path = ckpt_auroc.best_model_path
    best_loss_path = ckpt_loss.best_model_path
    
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS//2,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    print("TESTING BEST AUROC")
    trainer.test(model, test_dataloader, ckpt_path=best_auroc_path, weights_only=False)

    print("TESTING BEST LOSS")
    trainer.test(model, test_dataloader, ckpt_path=best_loss_path, weights_only=False)

if __name__ == "__main__":
    main()