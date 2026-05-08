import torch
import torch.onnx

from pathlib import Path

from src.utils import load_config, resolve_checkpoint_path
from src.model import DamageNet
from src.dataset import xBDDataset

xbd_config   = load_config('xbd.yaml')
model_config = load_config('model.yaml')

model       = DamageNet(config=model_config)
sample_data = xBDDataset(mode='train', stage=2, config=xbd_config)[0]

pre_tensor  = sample_data[xbd_config['item_group']['pre_image']].unsqueeze(0)
post_tensor = sample_data[xbd_config['item_group']['post_image']].unsqueeze(0)
dummy_input = (pre_tensor, post_tensor)

# Load Stage 2 checkpoint using the candidate path mechanism
model_path = resolve_checkpoint_path(model_config, stage=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Output path — write to configured onnx_dir, fall back to /kaggle/working
onnx_dir = Path(model_config['models']['onnx_dir'])
onnx_filename = 'damagenet.onnx'

if onnx_dir.exists():
    output_path = onnx_dir / onnx_filename
else:
    output_path = Path('/kaggle/working') / onnx_filename

print(f'Saving ONNX model to: {output_path}')

torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
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