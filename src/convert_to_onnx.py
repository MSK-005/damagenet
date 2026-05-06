import torch
import torch.onnx

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

model_file_name = 'stage2_best.pth'

try:
    model_file_path = get_file_path(filename=model_file_name, folders='models')
except FileNotFoundError:
    try:
        model_file_path = get_file_path(filename=model_file_name, folders='/kaggle/input/models/msk005/damagenet/pytorch/default/1')
    except FileNotFoundError:
        raise Exception('Could not find model. Please upload it in the data folder.' \
        'If you are on a separate Kaggle notebook just for converting the model, then upload the model on Kaggle as a dataset and import from there.')

model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
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