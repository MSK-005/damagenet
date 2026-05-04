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
from src.model import DamageNet

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

xbd_config = load_config('xbd.yaml')
model_config = load_config('model.yaml')

cfg = model_config['stage2']
num_classes = cfg['num_classes']

class_weights = torch.tensor([1.0, 4.0, 8.0])


def loss_fn(output, target, weights):
    dice = smp.losses.DiceLoss(mode='multiclass', classes=num_classes)
    focal = smp.losses.FocalLoss(mode='multiclass')
    ce = torch.nn.CrossEntropyLoss(weight=weights.to(output.device))
    return dice(output, target) + focal(output, target) + 2.0 * ce(output, target)


def compute_metrics_from_confusion(confusion):
    f1_per_class = []
    precision_per_class = []
    recall_per_class = []
    class_counts = confusion.sum(axis=1)
    total = class_counts.sum()

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    macro_f1 = np.mean(f1_per_class)
    weighted_f1 = np.sum([f1_per_class[i] * class_counts[i] for i in range(num_classes)]) / total

    return macro_f1, weighted_f1, np.mean(precision_per_class), np.mean(recall_per_class)


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader)):
        pre = batch['image'].to(device)
        post = batch['post_image'].to(device)
        target = batch['post_image_target'].to(device).long()

        with autocast('cuda'):
            output = model(pre, post)
            loss = loss_fn(output, target, class_weights) / accumulation_steps

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
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            pre = batch['image'].to(device)
            post = batch['post_image'].to(device)
            target = batch['post_image_target'].to(device).long()

            with autocast('cuda'):
                output = model(pre, post)
                loss = loss_fn(output, target, class_weights)

            total_loss += loss.item()

            preds = output.argmax(dim=1).cpu().numpy().flatten()
            targets = target.cpu().numpy().flatten()
            np.add.at(confusion, (targets, preds), 1)

            del output, pre, post, target
            torch.cuda.empty_cache()

    macro_f1, weighted_f1, precision, recall = compute_metrics_from_confusion(confusion)
    return total_loss / len(loader), macro_f1, weighted_f1, precision, recall


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(std_range=(0.01, 0.05)),
    ], p=0.3)
], additional_targets={
    'post_image': 'image',
    'pre_image_target': 'mask',
    'post_image_target': 'mask',
})

train_dataset = xBDDataset(mode='train', config=xbd_config, stage=2, transforms=train_transforms)
val_dataset = xBDDataset(mode='test', config=xbd_config, stage=2)

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

model = DamageNet(config=model_config).to(device)

print('Loading Stage 1 weights...')
stage1_state = torch.load('/kaggle/working/stage1_best.pth', map_location=device)
stage1_state = {k.replace('module.', ''): v for k, v in stage1_state.items()}

encoder_state = {
    k.replace('model.encoder.', ''): v
    for k, v in stage1_state.items()
    if k.startswith('model.encoder.')
}
model.encoder.load_state_dict(encoder_state, strict=False)
print(f'Loaded {len(encoder_state)} encoder layers from Stage 1.')

for param in model.encoder.parameters():
    param.requires_grad = False
print("Encoder frozen for initial Stage 2 training.")

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
best_macro_f1 = 0.0
epochs = cfg['epochs']

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    if epoch >= 3:
        for param in model.encoder.parameters():
            param.requires_grad = True

    train_loss = train_one_epoch(
        model, train_loader, optimizer,
        scaler, device, cfg['accumulation_steps'],
    )

    val_loss, macro_f1, weighted_f1, precision, recall = validate(
        model, val_loader, device,
    )

    scheduler.step()

    print(f'Train Loss:    {train_loss:.4f}')
    print(f'Val Loss:      {val_loss:.4f}')
    print(f'F1 (Macro):    {macro_f1:.4f}')
    print(f'F1 (Weighted): {weighted_f1:.4f}')
    print(f'Precision:     {precision:.4f}')
    print(f'Recall:        {recall:.4f}')

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        try:
            # If model was trained using parallel GPUs
            state_dict = model.module.state_dict()
        except AttributeError:
            # Model was not trained on parallel GPUs
            state_dict = model.state_dict()
        torch.save(state_dict, '/kaggle/working/stage2_best.pth')
        print(f'  Saved best model (Macro F1: {best_macro_f1:.4f})')

print(f'\nStage 2 complete. Best Macro F1: {best_macro_f1:.4f}')