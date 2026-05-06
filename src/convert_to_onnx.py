import torch
import torch.onnx

from pathlib import Path

from src.utils import load_config, get_file_path
from src.model import DamageNet
from src.dataset import xBDDataset

xbd_config   = load_config('xbd.yaml')
model_config = load_config('model.yaml')

model       = DamageNet(config=model_config)
sample_data = xBDDataset(mode='train', stage=2, config=xbd_config)[0]

pre_tensor  = sample_data[xbd_config['item_group']['pre_image']].unsqueeze(0)
post_tensor = sample_data[xbd_config['item_group']['post_image']].unsqueeze(0)
dummy_input = (pre_tensor, post_tensor)

model_file_name = 'damagenet.pth'

CANDIDATE_PATHS = [
    get_file_path(filename=model_file_name, folders='models'),
    Path(f'/kaggle/input/damagenet/pytorch/default/1/{model_file_name}'),
]

model_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)

if model_path is None:
    raise FileNotFoundError(
        f'Could not find {model_file_name}. Either place it in models/ or '
        'upload it as a Kaggle dataset and attach it to this notebook.'
    )

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    str(get_file_path(filename='damagenet.onnx', folders='models')),
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['pre', 'post'],
    output_names=['output'],
    dynamic_axes={
        'pre':    {0: 'batch_size'},
        'post':   {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)
print("Conversion complete!")