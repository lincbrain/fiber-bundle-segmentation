import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation tasks.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Balancing factor for positive and negative classes.
            gamma: Focusing parameter to reduce the impact of easy examples.
            reduction: Specifies the reduction to apply to the output ('mean' or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use logits for numerical stability

    def forward(self, preds, targets):
        """
        Args:
            preds: Predicted logits [B, 1, H, W].
            targets: Ground truth binary masks [B, 1, H, W].

        Returns:
            Focal Loss value.
        """
        # Compute BCE loss
        bce_loss = self.bce_loss(preds, targets)

        # Apply the sigmoid to get probabilities
        probs = torch.sigmoid(preds)

        # Compute pt (prob of the true class)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Compute the focal loss term
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (0, 2, 3)  # sum over batch and spatial dims

        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice_score.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = BinaryDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets.float())
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = BinaryDiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets.float())
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal
