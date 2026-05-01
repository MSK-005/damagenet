import torch
from torch.utils.data import DataLoader

from src.dataset import xBDDataset
from src.utils import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xbd_config = load_config('xbd.yaml')
model_config = load_config('model.yaml')

training_dataset = xBDDataset('stats', xbd_config)
train_dataloader = DataLoader(training_dataset, batch_size=model_config['training']['batch_size'], shuffle=False)

total_pixels = 0
total = torch.zeros(3).to(device)
total_squared = torch.zeros(3).to(device)

image_count = 0

for images in train_dataloader:
    pre_image = images['image'].to(device)
    post_image = images['post_image'].to(device)
    combined = torch.cat((pre_image, post_image), 0).to(device)

    total_pixels += combined.size(0) * combined.size(2) * combined.size(3)
    total += torch.sum(combined, dim=[0, 2, 3])
    total_squared += torch.sum(combined**2, dim=[0, 2, 3])
    
    image_count += combined.size(0)

mean = total / total_pixels
# Variance = average of squares - square of average
std = torch.sqrt((total_squared / total_pixels) - torch.square(mean))
print(f"Image count: {image_count}; mean: {mean}; std: {std}")