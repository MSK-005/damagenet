import os
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A

from src.utils import load_config
from src.dataset import xBDDataset
from src.model import LocalizationNet
from src.losses import Stage1Loss

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

xbd_config   = load_config('xbd.yaml')
model_config = load_config('model.yaml')

cfg      = model_config['stage1']
save_dir = model_config['models']['stage1_dir']
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / cfg['checkpoint']

wandb.init(
    project="damagenet",
    name="stage1",
    config={
        "encoder":            model_config['model']['name'],
        "epochs":             cfg['epochs'],
        "batch_size":         cfg['batch_size'],
        "learning_rate":      cfg['learning_rate'],
        "accumulation_steps": cfg['accumulation_steps'],
        "bce_weight":         cfg['bce_weight'],
        "dice_weight":        cfg['dice_weight'],
        "lovasz_weight":      cfg['lovasz_weight'],
        "pos_weight":         cfg['pos_weight'],
    }
)


def compute_binary_metrics(preds_sigmoid: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """
    Compute IoU and F1 for binary segmentation.
    preds_sigmoid: float array of sigmoid probabilities, shape (N,)
    targets:       binary int array, shape (N,)
    """
    preds = (preds_sigmoid > threshold).astype(np.int32)
    tp = np.logical_and(preds == 1, targets == 1).sum()
    fp = np.logical_and(preds == 1, targets == 0).sum()
    fn = np.logical_and(preds == 0, targets == 1).sum()

    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return float(iou), float(f1), float(precision), float(recall)


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader)):
        image  = batch['image'].to(device)
        target = batch['pre_image_target'].to(device).float().unsqueeze(1)

        with autocast('cuda'):
            output = model(image)
            loss   = loss_fn(output, target) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss    = 0.0
    all_preds     = []
    all_targets   = []

    with torch.no_grad():
        for batch in loader:
            image  = batch['image'].to(device)
            target = batch['pre_image_target'].to(device).float().unsqueeze(1)

            with autocast('cuda'):
                output = model(image)
                loss   = loss_fn(output, target)

            total_loss += loss.item()

            probs = torch.sigmoid(output).cpu().numpy().flatten()
            tgts  = target.cpu().numpy().flatten().astype(np.int32)
            all_preds.append(probs)
            all_targets.append(tgts)

            del output, image, target

    torch.cuda.empty_cache()

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    iou, f1, precision, recall = compute_binary_metrics(all_preds, all_targets)

    return total_loss / len(loader), iou, f1, precision, recall


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

loss_fn = Stage1Loss(
    pos_weight=torch.tensor([cfg['pos_weight']]).to(device),
    bce_weight=cfg['bce_weight'],
    dice_weight=cfg['dice_weight'],
    lovasz_weight=cfg['lovasz_weight'],
    clamp_range=20.0,
)

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomScale(scale_limit=(-0.5, 0.0), p=0.5),
    A.PadIfNeeded(
        min_height=1024,
        min_width=1024,
        border_mode=0,
        fill=0,
        fill_mask=0,
    ),
    A.RandomCrop(height=1024, width=1024),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.OneOf([
        A.GaussianBlur(sigma_limit=(3, 5)),
        A.GaussNoise(std_range=(0.01, 0.05)),
    ], p=0.3),
], additional_targets={
    xbd_config['item_group']['pre_image_target']: 'mask',
})

train_dataset = xBDDataset(mode='train', config=xbd_config, stage=1, transforms=train_transforms)
val_dataset   = xBDDataset(mode='test',  config=xbd_config, stage=1)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg['batch_size'],
    shuffle=True,
    num_workers=cfg['num_workers'],
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg['batch_size'],
    shuffle=False,
    num_workers=cfg['num_workers'],
    pin_memory=True,
    persistent_workers=True,
)

print(f'Training samples:   {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')

model = LocalizationNet(config=model_config).to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg['learning_rate'],
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg['epochs'],
)

scaler        = GradScaler('cuda')
best_val_loss = float('inf')
epochs        = cfg['epochs']

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    train_loss = train_one_epoch(
        model, train_loader, optimizer,
        scaler, device, cfg['accumulation_steps'],
    )

    val_loss, iou, f1, precision, recall = validate(model, val_loader, device)

    scheduler.step()

    print(f'Train Loss : {train_loss:.4f}')
    print(f'Val Loss   : {val_loss:.4f}')
    print(f'IoU        : {iou:.4f}')
    print(f'F1         : {f1:.4f}')
    print(f'Precision  : {precision:.4f}')
    print(f'Recall     : {recall:.4f}')

    wandb.log({
        "epoch":      epoch + 1,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "iou":        iou,
        "f1":         f1,
        "precision":  precision,
        "recall":     recall,
        "lr":         scheduler.get_last_lr()[0],
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'  Saved best model → {save_path}  (Val Loss: {best_val_loss:.4f})')

print(f'\nStage 1 complete. Best Val Loss: {best_val_loss:.4f}')
wandb.finish()