"""
train.py - 3D CNN Training for pCR Prediction
==============================================
 
Trains a 3D CNN to predict pathological complete 
response from DCE-MRI with PyTorch Lightning.
 
- Loading train / val / test splits from :class:`dataset.BreastDCEDataset`.
- Computing class weights from training set.
- Training :class:`PcrCNN` model with Adam, ReduceLROnPlateau scheduling, 
  and early stopping.
- Saves best-AUROC and best-loss checkpoints, then evaluates both on the
  test split.
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
"""str: Path to metadata CSV containing patient IDs, pCR labels,
and splits."""

DATAPATH = "./data"
"""str: Root directory of the three dataset folders."""

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 30
NUM_WORKERS = 8
PERSISTENT_WORKERS = bool(NUM_WORKERS)
SEED = 67

class ConvBlock(nn.Module):
    """Convolution block for the CNN.
    
    Builds a sequential layer of 3D convolution, batch
    normalization, ReLU activation, and max pooling.
    """
    
    def __init__(self, num_input_channels, num_output_channels):
        """
        :param num_input_channels: Number of input feature channels.
        :type num_input_channels: int
        :param num_output_channels: Number of output feature channels.
        :type num_output_channels: int
        """
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(num_input_channels, num_output_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(num_output_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2)
                                   )
        
    def forward(self, x):
        """Pass input through the layer.
        
        :param x: Input tensor of shape (N, C_in, D, H, W).
        :type x: torch.Tensor
        :returns: Output tensor of shape (N, C_out, D//2, H//2, W//2)
        :rtype: torch.Tensor
        """
        return self.block(x)

class PcrCNN(pl.LightningModule, nn.Module):
    """3D CNN for binary pCR classification.
    
    Given input of (N, 3, 32, 256, 256) - 3 DCE time points x 32 slices x 256x256 pixels:
    
    1. Encoder:
        - ConvBlock(3, 16) - (N, 16, 16, 128, 128)
        - ConvBlock(16, 32) - (N, 32, 8, 64, 64)
        - ConvBlock(32, 64) - (N, 64, 4, 32, 32)
        - ConvBlock(64, 128) - (N, 128, 2, 16, 16)
        - AdaptiveAvgPool3d - (N, 128, 1, 1, 1)
    
    2. Classifier:
        - Flatten (N, 128)
        - Linear(128, 128), ReLU, Dropout(0.5)
        - Linear(128, 1) - logit for BCEWithLogitsLoss
        
    Tracking AUROC of train / val / test.
    Tracking Accuracy of val / test.
    """
    def __init__(self, learning_rate=LEARNING_RATE, pos_weight=1.0):
        """
        :param learning_rate: Initial learning rate.
        :type learning_rate: float
        :param pos_weight: Weighted of positive pCR class in 
            :class:`~torch.nn.BCEWithLogitsLoss` to handle
            class imbalance.
        :type pos_weight: float
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
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
        """Run the encoder and classifier
        
        :param x: Batch of DCE-MRI volumes of shape (N, 3, 32, 256, 256).
        :type x: torch.Tensor
        :returns: Logits of shape (N, 1).
        :rtype: torch.Tensor
        """
        return self.classifier(self.encoder(x))

    def training_step(self, batch, batch_idx):
        """Return loss of one training batch and update AUROC.
        
        :param batch: Tuple of (images, labels) from training dataloader.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :returns: Training loss.
        :rtype: torch.Tensor
        """
        loss, probs, labels = self.shared_step(batch)
        
        self.train_auroc.update(probs, labels.int())
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Compute loss of one validation batch and update AUROC and accuracy
        
        :param batch: Tuple of (images, labels) from validation dataloader.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :returns: Validation loss.
        :rtype: torch.Tensor
        """
        loss, probs, labels = self.shared_step(batch)
        
        self.val_auroc.update(probs, labels.int())
        self.val_acc.update(probs, labels.int())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def shared_step(self, batch):
        """Forward pass and compute loss shared across train, val and test steps.
        
        :param batch: Tuple of (images, labels)
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :returns: Tuple of (loss, predicted probabilities, labels)
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        imgs, labels = batch
        logits = self(imgs).squeeze(1)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        return loss, probs, labels
    
    def on_train_epoch_end(self):
        """Log and reset training AUROC end of epoch."""
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        
        self.train_auroc.reset()
        
    def on_validation_epoch_end(self):
        """Log and reset validation AUROC and accuracy end of epoch"""
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        
        self.val_auroc.reset()
        self.val_acc.reset()
    
    def test_step(self, batch, batch_idx):
        """Compute loss for one test batch and update AUROC and accuracy
        
        :param batch: Tuple of (images, labels) from the test dataloader.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :param batch_idx: Index of batch
        :type batch_idx: int
        :returns: Test loss
        :rtype: torch.Tensor
        """
        loss, probs, labels = self.shared_step(batch)
        
        self.test_auroc.update(probs, labels.int())
        self.test_acc.update(probs, labels.int())
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Log and reset test AUROC and accuracy end of epoch."""
        self.log("test_auroc", self.test_auroc.compute(), prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        
        self.test_auroc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        """Set up Adam optimizer with weight decay and ReduceLROnPlateau
        scheduler. The scheduler monitors val_auroc and halves learning rate
        after 5 epochs without improvement.
        
        :returns: Dict with optimizer and lr_scheduler keys
        :rtype: dict
        """
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
        """Checks for NaN weights at the end of every training batch.
        
        :param outputs: Return value of :meth:`training_step`.
        :type outputs: torch.Tensor
        :param batch: Current batch of (images, labels)
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :param batch_idx: Index of batch.
        :type batch_idx: int
        """
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"\n[CRITICAL WARNING] NaN weights detected in layer: {name} at batch {batch_idx}!")
                # Optional: break to stop spamming the console
                break

def main():
    """Train and evaluate by:
    
    - Seed random number generators with :data:`SEED`.
    - Loads train, validation, and test splits.
    - Computes pos_weight from training label distribution
    - Trains :class:`PcrCNN` and checkpointing best AUROC and
      validation loss with early stopping on AUROC with patience.
    - Exports model_best_auroc.pth and model_best_loss.pth.
    - Evaluate both checkpoints on test split.
    """
    pl.seed_everything(SEED)

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
    
    model = PcrCNN(learning_rate=LEARNING_RATE, pos_weight=pos_weight)
    
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

    best_auroc = PcrCNN.load_from_checkpoint(ckpt_auroc.best_model_path, weights_only=False)
    torch.save(best_auroc.state_dict(), "model_best_auroc.pth")

    best_loss = PcrCNN.load_from_checkpoint(ckpt_loss.best_model_path, weights_only=False)
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