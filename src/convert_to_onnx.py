import torch
import torch.onnx

from src.utils import load_config, get_file_path
from src.model import DamageNet
from src.dataset import xBDDataset

xbd_config = load_config('xbd.yaml')
model_config = load_config('model.yaml')
model = DamageNet(config=model_config)
sample_data = xBDDataset(mode='train', stage=2, config=xbd_config)[0]

pre_tensor = sample_data[xbd_config['item_group']['pre_image']].unsqueeze(0)
post_tensor = sample_data[xbd_config['item_group']['post_image']].unsqueeze(0)
dummy_input = (pre_tensor, post_tensor)
model_path = get_file_path(filename='model.pth', folders='models')

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

torch.onnx.export(
    model, 
    dummy_input, 
    str(get_file_path(filename='damagenet.onnx', folders='models')), 
    export_params=True, 
    opset_version=18,
    input_names=['pre', 'post'], 
    output_names=['output']
)
print("Conversion complete!")
