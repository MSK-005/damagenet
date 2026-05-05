from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import numpy as np
import albumentations as A
from src.utils import get_xbd_image_ids


REMAP_5_TO_3 = {
    0: 0,  # background     -> no damage
    1: 0,  # no damage      -> no damage
    2: 1,  # minor damage   -> partial damage
    3: 1,  # major damage   -> partial damage
    4: 2,  # destroyed      -> destroyed
}


def remap_mask(mask):
    remapped = np.zeros_like(mask)
    for original, new in REMAP_5_TO_3.items():
        remapped[mask == original] = new
    return remapped


class xBDDataset(Dataset):
    def __init__(self, mode, config, stage=2, transforms=None):
        mode_to_dir = {
            'train': 'train',
            'test':  'test',
            'stats': 'train',
        }

        if mode not in mode_to_dir:
            raise Exception(f'mode should be in {[*mode_to_dir]}')
        elif mode == 'stats' and transforms is not None:
            raise Exception(f'You cannot apply transformations to the data if your mode is \'{mode}\'.')

        self.mode = mode
        self.stage = stage
        self.config = config

        self.root_dir = self.config[mode_to_dir[mode]]['abs_path']
        if not self.root_dir.exists():
            raise Exception(f'Could not find path: {self.root_dir}')

        self.ids = get_xbd_image_ids(self.root_dir / self.config['folders']['images'])
        self.transforms = transforms

        transform_img_keys = self.config['item_group']

        transforms_list = [A.ToFloat(max_value=255.0)]
        additional_targets = {}

        if self.mode != 'stats':
            transforms_list.append(
                A.Normalize(
                    mean=self.config['stats']['mean'],
                    std=self.config['stats']['std'],
                )
            )

            if self.stage == 1:
                additional_targets[transform_img_keys['pre_image_target']] = 'mask'
            else:
                additional_targets[transform_img_keys['post_image']] = 'image'
                additional_targets[transform_img_keys['pre_image_target']] = 'mask'
                additional_targets[transform_img_keys['post_image_target']] = 'mask'

        transforms_list.append(ToTensorV2())

        self.convert_to_tensor = A.Compose(
            transforms_list,
            additional_targets=additional_targets,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        group = self.config['item_group']
        img_dir = self.root_dir / self.config['folders']['images']
        target_dir = self.root_dir / self.config['folders']['targets']
        suffix = self.config['naming']
        ext = self.config['naming']['extension']

        # Pre-disaster image
        pre_file = f"{image_id}{suffix['pre_suffix']}{ext}"
        image = np.array(Image.open(img_dir / pre_file).convert('RGB'))

        # Pre-disaster target (building footprint for stage 1, or localization for stage 2)
        pre_target_file  = f"{image_id}{suffix['pre_target_suffix']}{ext}"
        pre_image_target = np.array(Image.open(target_dir / pre_target_file).convert('L'))
        pre_image_target = (pre_image_target > 0).astype(np.uint8)
        
        print(np.unique(pre_image_target))

        if self.mode == 'stats':
            data = {group['pre_image']: image}
            data = self.convert_to_tensor(**data)
            return data

        if self.stage == 1:
            data = {
                group['pre_image']: image,
                group['pre_image_target']: pre_image_target,
            }

            if self.transforms:
                data = self.transforms(**data)

            data = self.convert_to_tensor(**data)
            return data

        # Stage 2 — load post image and post target
        post_file = f"{image_id}{suffix['post_suffix']}{ext}"
        post_image = np.array(Image.open(img_dir / post_file).convert('RGB'))

        post_target_file = f"{image_id}{suffix['post_target_suffix']}{ext}"
        post_image_target = np.array(Image.open(target_dir / post_target_file).convert('L'))

        post_image_target = remap_mask(post_image_target)

        data = {
            group['pre_image']: image,
            group['post_image']: post_image,
            group['pre_image_target']: pre_image_target,
            group['post_image_target']: post_image_target,
        }

        if self.transforms:
            data = self.transforms(**data)

        data = self.convert_to_tensor(**data)
        return data

    def __repr__(self):
        return f'DamageNet (root: {self.root_dir}, stage: {self.stage}, samples: {len(self.ids)})'