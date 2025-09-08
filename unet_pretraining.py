"""
PretrainingModule for U-Net pretraining.
Takes a model and trains it to reconstruct input patches.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np 

class PretrainingModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # MSE loss for reconstruction
        self.loss_function = nn.MSELoss()
        
        # Add a 1->3 channel adaptation layer
        self.channel_adapter = nn.Conv2d(
            in_channels=1,    # Your model's output channels
            out_channels=3,   # Target channels
            kernel_size=1,    # 1x1 convolution
            bias=True
        )

    def forward(self, x):
        # Get the single-channel output from your model
        features = self.model(x)
        # Convert to 3 channels
        out = self.channel_adapter(features)
        return out

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        reconstruction = self(x)
        loss = self.loss_function(reconstruction, x)

        # Log loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, means, stds = batch

        reconstruction = self(x)
        loss = self.loss_function(reconstruction, x)
        
        # Log validation loss
        self.log('val_loss', loss, prog_bar=True)
        
        if batch_idx == 0:
            self._log_reconstructions(x, reconstruction, means, stds, prefix='val_')

    def _log_reconstructions(self, original, reconstruction, means, stds, prefix=''):
        # Take first image from batch
        original = original[0]
        reconstruction = reconstruction[0]

        means = means[0]  # Get stats for first image
        stds = stds[0]

        # Denormalize images
        original_img = self.denormalize_image(original.cpu().numpy(), means.cpu().numpy(), stds.cpu().numpy())
        recon_img = self.denormalize_image(reconstruction.cpu().numpy(), means.cpu().numpy(), stds.cpu().numpy())
        
        # Log to wandb
        self.logger.experiment.log({
            f"{prefix}original": wandb.Image(original_img, caption="Original"),
            f"{prefix}reconstruction": wandb.Image(recon_img, caption="Reconstruction")
        })
        
    def denormalize_image(self, normalized_image, means, stds):
        """
        Denormalize image using per-image statistics
        """
        normalized_image = normalized_image.transpose(1, 2, 0)
        denormalized = np.zeros_like(normalized_image, dtype=np.float32)
        for c in range(3):
            denormalized[:, :, c] = (normalized_image[:, :, c] * stds[c]) + means[c]
        return np.clip(denormalized, 0, 255).astype(np.uint8)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
