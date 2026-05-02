import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os   
from tqdm import tqdm

from src.dataset import xBDDataset
from src.model import DamageNet
from src.utils import load_config

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

xbd_config = load_config('xbd.yaml')
model_config = load_config('model.yaml')


def compute_metrics_from_confusion(confusion, num_classes):
    f1_per_class = []
    precision_per_class = []
    recall_per_class = []

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

    return (
        np.mean(f1_per_class),
        np.mean(precision_per_class),
        np.mean(recall_per_class)
    )


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader)):
        pre = batch['image'].to(device)
        post = batch['post_image'].to(device)
        target = batch['post_image_target'].to(device).long()

        with autocast('cuda'):
            output = model(pre, post)
            loss = loss_fn(output, target) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            pre = batch["image"].to(device)
            post = batch["post_image"].to(device)
            target = batch["post_image_target"].to(device).long()

            with autocast("cuda"):
                output = model(pre, post)
                loss = loss_fn(output, target)

            total_loss += loss.item()

            preds = output.argmax(dim=1).cpu().numpy().flatten()
            targets = target.cpu().numpy().flatten()

            np.add.at(confusion, (targets, preds), 1)

            del output, pre, post, target
            torch.cuda.empty_cache()

    f1, precision, recall = compute_metrics_from_confusion(confusion, num_classes)
    return total_loss / len(loader), f1, precision, recall


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset = xBDDataset(mode='train', config=xbd_config)
val_dataset = xBDDataset(mode='test', config=xbd_config)

train_loader = DataLoader(
    train_dataset,
    batch_size=model_config['training']['batch_size'],
    shuffle=True,
    num_workers=model_config['training']['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=model_config['training']['batch_size'],
    shuffle=False,
    num_workers=model_config['training']['num_workers'],
    pin_memory=True
)

print(f'Training samples:   {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')

model = DamageNet(config=model_config).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')


num_classes = model_config['classes']['num_classes']

dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=num_classes)
focal_loss = smp.losses.FocalLoss(mode='multiclass')

def loss_fn(output, target):
    return dice_loss(output, target) + focal_loss(output, target)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=model_config['training']['learning_rate'],
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=model_config['training']['epochs']
)

scaler = GradScaler('cuda')
accumulation_steps = model_config['training']['accumulation_steps']

best_f1 = 0.0
epochs = model_config['training']['epochs']

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    train_loss = train_one_epoch(
        model, train_loader, optimizer, loss_fn,
        scaler, device, accumulation_steps
    )

    val_loss, f1, precision, recall = validate(
        model, val_loader, loss_fn, device, num_classes
    )

    scheduler.step()

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss:   {val_loss:.4f}')
    print(f'F1:         {f1:.4f}')
    print(f'Precision:  {precision:.4f}')
    print(f'Recall:     {recall:.4f}')

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  Saved best model (F1: {best_f1:.4f})')

print(f'\nTraining complete. Best F1: {best_f1:.4f}')
torch.save(model.state_dict(), 'model_weights.pth')