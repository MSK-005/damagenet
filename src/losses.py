import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class StableBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        return self.bce(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)

        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class Stage1Loss(nn.Module):
    """
    BCEWithLogits + Dice + Lovasz (binary segmentation)
    """

    def __init__(
        self,
        pos_weight=None,
        bce_weight=1.0,
        dice_weight=1.0,
        lovasz_weight=1.0,
        clamp_logits=True
    ):
        super().__init__()

        self.bce = StableBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.lovasz = smp.losses.LovaszLoss(mode='binary')

        self.w_bce = bce_weight
        self.w_dice = dice_weight
        self.w_lovasz = lovasz_weight

        self.clamp_logits = clamp_logits

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        if self.clamp_logits:
            logits = torch.clamp(logits, -10, 10)

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)

        return (
            self.w_bce * bce_loss +
            self.w_dice * dice_loss +
            self.w_lovasz * lovasz_loss
        )


class Stage2Loss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()

        self.register_buffer("class_weights", class_weights)

        self.dice = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True
        )

        self.focal = smp.losses.FocalLoss(
            mode='multiclass',
            normalized=True
        )

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logits, target):
        logits = logits.float()

        loss_dice = self.dice(logits, target)
        loss_focal = self.focal(logits, target)
        loss_ce = self.ce(logits, target)

        return loss_dice + loss_focal + 2.0 * loss_ce