import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

IGNORE_INDEX = 255


class StableBCEWithLogitsLoss(nn.Module):
    """
    Wraps nn.BCEWithLogitsLoss and forces fp32 computation.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.float(), targets.float())


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation, computed in fp32.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.float()
        targets = targets.float()

        probs   = torch.sigmoid(logits)
        probs   = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)
        dice         = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class Stage1Loss(nn.Module):
    """
    Binary building localization loss: BCE + Dice + Lovász.
    """

    def __init__(
        self,
        pos_weight=None,
        bce_weight:    float = 1.0,
        dice_weight:   float = 1.0,
        lovasz_weight: float = 1.0,
        clamp_logits:  bool  = True,
        clamp_range:   float = 20.0,
    ):
        super().__init__()

        self.bce    = StableBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice   = DiceLoss()
        self.lovasz = smp.losses.LovaszLoss(mode='binary', from_logits=True)

        self.w_bce    = bce_weight
        self.w_dice   = dice_weight
        self.w_lovasz = lovasz_weight

        self.clamp_logits = clamp_logits
        self.clamp_range  = clamp_range

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.float()
        targets = targets.float()

        if self.clamp_logits:
            logits = torch.clamp(logits, -self.clamp_range, self.clamp_range)

        bce_loss    = self.bce(logits, targets)
        dice_loss   = self.dice(logits, targets)
        lovasz_loss = self.lovasz(logits, targets.long())

        return (
            self.w_bce    * bce_loss    +
            self.w_dice   * dice_loss   +
            self.w_lovasz * lovasz_loss
        )


class Stage2Loss(nn.Module):
    """
    Multi-class damage classification loss: Dice + Focal + CE.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()

        self.register_buffer('class_weights', class_weights.float())

        self.dice = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True,
            ignore_index=IGNORE_INDEX,
        )
        self.focal = smp.losses.FocalLoss(
            mode='multiclass',
            normalized=True,
            ignore_index=IGNORE_INDEX,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()

        loss_dice  = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)
        loss_ce    = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype),
            ignore_index=IGNORE_INDEX,
        )

        return loss_dice + loss_focal + 2.0 * loss_ce