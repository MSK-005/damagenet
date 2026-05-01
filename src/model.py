import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict

class DamageNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=config["model"]["name"],
            encoder_weights=config["model"]["encoder_weights"],
            in_channels=config["model"]["in_channels"],
            classes=config["classes"]["num_classes"],
            activation=None,
            decoder_attention_type=config["model"]["attention_type"]
        )    
    
    def forward(self, pre, post):
        combined = torch.cat((pre, post), 1)
        output = self.model(combined)
        return output
