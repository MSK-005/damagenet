import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class StableBCEWithLogitsLoss(nn.Module):
    """
    Wraps nn.BCEWithLogitsLoss and forces fp32 computation.
    autocast can emit fp16 logits; BCEWithLogitsLoss is numerically
    safe in fp32 but can overflow/underflow in fp16.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.float(), targets.float())


class DiceLoss(nn.Module):
    """
    Soft Dice loss computed in fp32.
    eps in both numerator and denominator prevents 0/0 on empty masks.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force fp32 — cumulative sums in fp16 lose precision fast
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
    Binary segmentation loss: BCE + Dice + Lovász.

    Inspired by the xView2 winning solution.  All three terms are
    computed in fp32 so autocast (fp16 forward pass) cannot produce NaN.

    Key differences from the original:
    - clamp range widened to [-20, 20] (+-10 zeros gradients at init)
    - LovaszLoss receives fp32 logits via explicit cast before entry
    - pos_weight tensor stays float32 regardless of autocast dtype
    """

    def __init__(
        self,
        pos_weight=None,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        clamp_logits: bool = True,
        clamp_range: float = 20.0,
    ):
        super().__init__()

        self.bce = StableBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        # smp LovaszLoss (binary) — receives logits, handles sigmoid internally
        self.lovasz = smp.losses.LovaszLoss(mode='binary', from_logits=True)

        self.w_bce    = bce_weight
        self.w_dice   = dice_weight
        self.w_lovasz = lovasz_weight

        self.clamp_logits = clamp_logits
        self.clamp_range  = clamp_range

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 once here — every loss term is safe
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

    Inspired by the xView2 winning solution.  Three fixes vs. original:

    1. FocalLoss receives raw logits — passing pre-softmaxed probabilities
       caused log(p) to underflow to -inf in fp16.
    2. CrossEntropyLoss weight is cast to fp32 + correct device inside
       forward, fixing DataParallel replicas that receive fp16 inputs.
    3. All terms explicitly cast logits to fp32 before computation.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()

        self.register_buffer('class_weights', class_weights.float())

        self.dice = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True,
        )
        # normalized=True uses log_softmax internally — numerically stable
        self.focal = smp.losses.FocalLoss(
            mode='multiclass',
            normalized=True,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 cast protects all three terms under autocast
        logits = logits.float()

        loss_dice  = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)
        # Explicit device + dtype cast is critical for DataParallel
        loss_ce    = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype),
        )

        return loss_dice + loss_focal + 2.0 * loss_ce
