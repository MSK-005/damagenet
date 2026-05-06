import os
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A

from src.utils import load_config, resolve_checkpoint_path
from src.dataset import xBDDataset
from src.model import DamageNet
from src.losses import Stage2Loss

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

xbd_config   = load_config('xbd.yaml')
model_config = load_config('model.yaml')

cfg      = model_config['stage2']
save_dir = model_config['models']['stage2_dir']
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / cfg['checkpoint']

num_classes            = cfg['num_classes']
encoder_unfreeze_epoch = cfg['encoder_unfreeze_epoch']
class_weights          = torch.tensor(cfg['class_weights'])

wandb.init(
    project="damagenet",
    name="stage2",
    config={
        "encoder":                model_config['model']['name'],
        "epochs":                 cfg['epochs'],
        "batch_size":             cfg['batch_size'],
        "learning_rate":          cfg['learning_rate'],
        "accumulation_steps":     cfg['accumulation_steps'],
        "num_classes":            num_classes,
        "class_weights":          cfg['class_weights'],
        "encoder_unfreeze_epoch": encoder_unfreeze_epoch,
    }
)

CLASS_NAMES = ['no_damage', 'partial_damage', 'destroyed']


def compute_metrics_from_confusion(confusion):
    """
    Returns per-class and aggregate metrics from a confusion matrix.
    Excludes classes absent from both predictions and targets (IoU undefined).
    """
    f1_per_class        = []
    iou_per_class       = []
    precision_per_class = []
    recall_per_class    = []
    class_counts        = confusion.sum(axis=1)
    total               = class_counts.sum()

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        iou_per_class.append(iou)

    macro_f1  = float(np.mean(f1_per_class))
    macro_iou = float(np.mean(iou_per_class))

    return (
        macro_f1,
        macro_iou,
        f1_per_class,
        iou_per_class,
        precision_per_class,
        recall_per_class,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader)):
        pre    = batch['image'].to(device)
        post   = batch['post_image'].to(device)
        target = batch['post_image_target'].to(device).long()

        with autocast('cuda'):
            output = model(pre, post)
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
    total_loss = 0.0
    confusion  = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            pre    = batch['image'].to(device)
            post   = batch['post_image'].to(device)
            target = batch['post_image_target'].to(device).long()

            with autocast('cuda'):
                output = model(pre, post)
                loss   = loss_fn(output, target)

            total_loss += loss.item()

            preds   = output.argmax(dim=1).cpu().numpy().flatten()
            targets = target.cpu().numpy().flatten()

            valid   = targets != 255
            np.add.at(confusion, (targets[valid], preds[valid]), 1)

            del output, pre, post, target

    torch.cuda.empty_cache()

    macro_f1, macro_iou, f1_per_class, iou_per_class, precision_per_class, recall_per_class = \
        compute_metrics_from_confusion(confusion)

    return (
        total_loss / len(loader),
        macro_f1,
        macro_iou,
        f1_per_class,
        iou_per_class,
        precision_per_class,
        recall_per_class,
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

loss_fn = Stage2Loss(class_weights=class_weights).to(device)

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
    xbd_config['item_group']['post_image']:        'image',
    xbd_config['item_group']['pre_image_target']:  'mask',
    xbd_config['item_group']['post_image_target']: 'mask',
})

train_dataset = xBDDataset(mode='train', config=xbd_config, stage=2, transforms=train_transforms)
val_dataset   = xBDDataset(mode='test',  config=xbd_config, stage=2)

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

model = DamageNet(config=model_config).to(device)

stage1_path = resolve_checkpoint_path(model_config, stage=1)
stage1_state = torch.load(stage1_path, map_location=device)
encoder_state = {
    k.replace('model.encoder.', ''): v
    for k, v in stage1_state.items()
    if k.startswith('model.encoder.')
}
model.encoder.load_state_dict(encoder_state, strict=False)
print(f'Loaded {len(encoder_state)} encoder layers from Stage 1.')

for param in model.encoder.parameters():
    param.requires_grad = False
print('Encoder frozen for initial Stage 2 training.')

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
best_macro_f1 = 0.0
best_val_loss = float('inf')
encoder_unfrozen = False
epochs        = cfg['epochs']

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    if epoch == encoder_unfreeze_epoch and not encoder_unfrozen:
        try:
            encoder = model.module.encoder
        except AttributeError:
            encoder = model.encoder
        for param in encoder.parameters():
            param.requires_grad = True
        encoder_unfrozen = True

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'] - encoder_unfreeze_epoch,
        )
        print(f'Encoder unfrozen. LR schedule restarted over {cfg["epochs"] - encoder_unfreeze_epoch} epochs.')

    train_loss = train_one_epoch(
        model, train_loader, optimizer,
        scaler, device, cfg['accumulation_steps'],
    )

    val_loss, macro_f1, macro_iou, f1_per_class, iou_per_class, precision_per_class, recall_per_class = \
        validate(model, val_loader, device)

    scheduler.step()

    print(f'Train Loss  : {train_loss:.4f}')
    print(f'Val Loss    : {val_loss:.4f}')
    print(f'F1  (Macro) : {macro_f1:.4f}')
    print(f'IoU (Macro) : {macro_iou:.4f}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:15s} — F1: {f1_per_class[i]:.4f}  IoU: {iou_per_class[i]:.4f}  '
              f'P: {precision_per_class[i]:.4f}  R: {recall_per_class[i]:.4f}')

    log_dict = {
        "epoch":      epoch + 1,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "f1_macro":   macro_f1,
        "iou_macro":  macro_iou,
        "lr":         scheduler.get_last_lr()[0],
    }
    for i, name in enumerate(CLASS_NAMES):
        log_dict[f'f1_{name}']        = f1_per_class[i]
        log_dict[f'iou_{name}']       = iou_per_class[i]
        log_dict[f'precision_{name}'] = precision_per_class[i]
        log_dict[f'recall_{name}']    = recall_per_class[i]

    wandb.log(log_dict)

    improved = macro_f1 > best_macro_f1 or (not encoder_unfrozen and val_loss < best_val_loss)
    if improved:
        best_macro_f1 = max(macro_f1, best_macro_f1)
        best_val_loss = min(val_loss, best_val_loss)
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'  Saved best model → {save_path}  (Macro F1: {best_macro_f1:.4f})')

print(f'\nStage 2 complete. Best Macro F1: {best_macro_f1:.4f}')
wandb.finish()