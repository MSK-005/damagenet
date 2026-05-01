from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2

import numpy as np
import albumentations as A

from src.utils import get_xbd_image_ids

class xBDDataset(Dataset):
    def __init__(self, mode, config, transforms=None):
        mode_to_dir = {
            'train': 'train',
            'test': 'test',
            'stats': 'train'
        }

        if mode not in mode_to_dir:
            raise Exception(f'mode should be in {[*mode_to_dir]}')
        elif mode == 'stats' and transforms is not None:
            raise Exception(f'You cannot apply transformations to the data if your mode is \'{mode}\'.')
        else:
            self.mode = mode
        
        self.config = config
        self.root_dir = self.config[mode_to_dir[mode]]['abs_path']
        if not self.root_dir.exists():
            raise Exception(f'Could not find path: {self.root_dir}')

        # TODO: once we expand to multiple datasets, we must change the function name, or the approach to get the ids
        self.ids = get_xbd_image_ids(self.root_dir / self.config['folders']['images'])
        self.transforms = transforms

        transform_img_keys = self.config['item_group']
        transforms_list = [A.ToFloat(max_value=255.0)]
        additional_targets = { transform_img_keys['post_image']: 'image' }
        if self.mode != 'stats':
            transforms_list.append(A.Normalize(mean=self.config['stats']['mean'], std=self.config['stats']['std']))
            additional_targets.update({
                transform_img_keys['pre_image_target']: 'mask',
                transform_img_keys['post_image_target']: 'mask'
            })            
        
        transforms_list.append(ToTensorV2())
        self.convert_to_tensor = A.Compose(transforms_list, additional_targets=additional_targets)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        image_id = self.ids[idx]
        group = self.config['item_group']
        img_dir = self.root_dir / self.config['folders']['images']
        target_dir = self.root_dir / self.config['folders']['targets']
        suffix_name = self.config['naming']
        ext = self.config['naming']['extension']

        # Pre-disaster image file
        suffix = f'{suffix_name['pre_suffix']}{ext}'
        file_name = f'{image_id}{suffix}'
        image = Image.open(img_dir / file_name).convert('RGB')
        image = np.array(image)

        # Post-disaster image file
        suffix = f'{suffix_name['post_suffix']}{ext}'
        file_name = f'{image_id}{suffix}'
        post_image = Image.open(img_dir / file_name).convert('RGB')
        post_image = np.array(post_image)

        # data only includes the base pre- and post-disaster images for computing statistics
        data = {
            group['pre_image']: image,
            group['post_image']: post_image
        }

        if self.mode == 'stats':
            data = self.convert_to_tensor(**data)
            return data

        # Target pre-disaster image file
        suffix = f'{suffix_name['pre_target_suffix']}{ext}'
        file_name = f'{image_id}{suffix}'
        pre_image_target = Image.open(target_dir / file_name).convert('L')
        pre_image_target = np.array(pre_image_target)

        # Target post-disaster image file
        suffix = f'{suffix_name['post_target_suffix']}{ext}'
        file_name = f'{image_id}{suffix}'
        post_image_target = Image.open(target_dir / file_name).convert('L')
        post_image_target = np.array(post_image_target)

        # data now includes the targets as well, since the dataset must return the base images, as well as the corresponding targets
        data[group['pre_image_target']] = pre_image_target
        data[group['post_image_target']] = post_image_target

        if self.transforms:
            data = self.transforms(**data)
        
        data = self.convert_to_tensor(**data)
        return data

    def __repr__(self):
        return f'DamageNet (root: {self.root_dir}, samples: {len(self.ids)})'
