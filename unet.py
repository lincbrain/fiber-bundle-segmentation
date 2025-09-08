import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loss_functions import DiceFocalLoss, DiceBCELoss
from torchmetrics.classification import BinaryJaccardIndex  # For IoU
from torchmetrics.classification import BinaryF1Score  # For Dice Score
import numpy as np

def denormalize_image(normalized_image):
    """
    Convert from Z-score normalized space back to 0-255 range
    """
    means = [26.370181406062926, 28.796625947529606, 25.990918044351993]
    stds = [31.776848516250062, 28.548426642727726, 16.703267170423146]
    
    denormalized = np.zeros_like(normalized_image, dtype=np.float32)
    for c in range(3):
        denormalized[:, :, c] = (normalized_image[:, :, c] * stds[c]) + means[c]
    
    # Clip to valid range and convert to uint8
    denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
    return denormalized

class DoubleConv_Flexible(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FlexibleUNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, min_features=32, max_features=512, loss="BCEdice"):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss
        
        # Features per stage
        self.features = [
            min(max_features, min_features * (2**i)) 
            for i in range(9)
        ]
        
        # Ensure last few stages maintain max_features
        for i in range(-4, 0):
            self.features[i] = max_features
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_channels = 3  # Input channels
        
        for feature in self.features:
            self.encoder_blocks.append(DoubleConv_Flexible(in_channels, feature))
            in_channels = feature
            
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(len(self.features)-1, 0, -1):
            # For upconv, we use the number of features from the current level
            self.upconvs.append(
                nn.ConvTranspose2d(self.features[i], self.features[i-1], kernel_size=2, stride=2)
            )
            # For decoder conv, we use twice the number of features due to concatenation
            self.decoder_blocks.append(
                DoubleConv_Flexible(self.features[i-1] * 2, self.features[i-1])  # Changed this line
            )
        
        # Final layer
        self.final_conv = nn.Conv2d(self.features[0], 1, kernel_size=1)
        
        if self.loss == "BCEdice":
            self.criterion = DiceBCELoss()
        elif self.loss == "focaldice":
            self.criterion = DiceFocalLoss()
        else:
            raise ValueError(f"Choose another loss!")

        # Metrics
        self.metric_iou = BinaryJaccardIndex()  # Intersection over Union
        self.metric_dice = BinaryF1Score()  # Dice Score

    def forward(self, x):
        # Check input dimensions
        h, w = x.shape[2:]
        num_pools = 0
        
        # Calculate how many pooling operations we can do
        curr_h, curr_w = h, w
        for i in range(len(self.features)-1):  # -1 because we don't pool after last encoder
            if curr_h >= 2 and curr_w >= 2:
                curr_h, curr_w = curr_h // 2, curr_w // 2
                num_pools += 1
            else:
                break
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path with dynamic pooling
        x = self.encoder_blocks[0](x)
        encoder_outputs.append(x)
        
        for i in range(1, num_pools + 1):
            x = F.max_pool2d(x, 2)
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)
        
        # Decoder path
        encoder_outputs = encoder_outputs[:-1]  # Remove last encoder output
        encoder_outputs = encoder_outputs[::-1]  # Reverse for easier indexing
        
        # Only use as many decoder blocks as we have encoder features
        for i in range(len(encoder_outputs)):
            x = self.upconvs[i](x)
            
            # Handle cases where dimensions don't match exactly
            if x.shape[2:] != encoder_outputs[i].shape[2:]:
                x = F.interpolate(x, encoder_outputs[i].shape[2:], mode='bilinear', align_corners=True)
            
            # Concatenate skip connection
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            x = self.decoder_blocks[i](x)
        
        return self.final_conv(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Add a channel dimension to masks if needed
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        y = y.float()

        loss = self.criterion(y_hat, y)

        # Convert logits to binary predictions
        preds = torch.sigmoid(y_hat) > 0.5  # Threshold at 0.5
            
        # Compute metrics
        iou = self.metric_iou(preds, y.int())
        dice = self.metric_dice(preds, y.int())

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_iou", iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice", dice, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Add a channel dimension to masks if needed
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        y = y.float()

        loss = self.criterion(y_hat, y)

        # Convert logits to binary predictions
        preds = torch.sigmoid(y_hat) > 0.5  # Threshold at 0.5

        # Compute metrics
        iou = self.metric_iou(preds, y.int())
        dice = self.metric_dice(preds, y.int())

        # Log metrics
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True, logger=True) 
        self.log("val_dice", dice, on_epoch=True, prog_bar=True, logger=True)

        
        if batch_idx == 0:
            predicted_image = (preds[0].cpu().numpy() * 255).astype('uint8')
            ground_truth_image = (y[0].cpu().numpy() * 255).astype('uint8')
            
            # Just transpose the channels, but don't multiply by 255
            input_image = x[0].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize using the statistics
            denormalized_image = denormalize_image(input_image)
            
            self.logger.experiment.log({
                "original_image": wandb.Image(denormalized_image, caption="Input"),
                "predicted_mask": wandb.Image(predicted_image, caption="Predicted Mask"),
                "ground_truth": wandb.Image(ground_truth_image, caption="Ground Truth")
            })

