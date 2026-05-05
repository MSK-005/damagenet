import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A

from src.utils import load_config
from src.dataset import xBDDataset
from src.model import LocalizationNet

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

xbd_config = load_config('xbd.yaml')
model_config = load_config('model.yaml')

cfg = model_config['stage1']


def loss_fn(output, target):
    dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    focal = smp.losses.FocalLoss(mode='binary', from_logits=True)
    return dice(output, target) + focal(output, target)


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader)):
        image = batch['image'].to(device)
        target = batch['pre_image_target'].to(device).float().unsqueeze(1)

        with autocast('cuda'):
            output = model(image)

        loss = loss_fn(output, target) / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            target = batch['pre_image_target'].to(device).float().unsqueeze(1)

            with autocast('cuda'):
                output = model(image)

            loss = loss_fn(output, target)
            total_loss += loss.item()
            del output, image, target

    torch.cuda.empty_cache()
    return total_loss / len(loader)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
    xbd_config['item_group']['pre_image_target']: 'mask'
})


train_dataset = xBDDataset(mode='train', config=xbd_config, stage=1, transforms=train_transforms)
val_dataset = xBDDataset(mode='test', config=xbd_config, stage=1)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg['batch_size'],
    shuffle=True,
    num_workers=cfg['num_workers'],
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg['batch_size'],
    shuffle=False,
    num_workers=cfg['num_workers'],
    pin_memory=True,
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

scaler = GradScaler('cuda')
best_val_loss = float('inf')
epochs = cfg['epochs']

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    train_loss = train_one_epoch(
        model, train_loader, optimizer,
        scaler, device, cfg['accumulation_steps'],
    )

    val_loss = validate(model, val_loader, device)

    scheduler.step()

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss:   {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        try:
            # If model was trained using parallel GPUs
            state_dict = model.module.state_dict()
        except AttributeError:
            # Model was not trained on parallel GPUs
            state_dict = model.state_dict()
        torch.save(state_dict, '/kaggle/working/stage1_best.pth')
        print(f'  Saved best Stage 1 model (Val Loss: {best_val_loss:.4f})')

print(f'\nStage 1 complete. Best Val Loss: {best_val_loss:.4f}')