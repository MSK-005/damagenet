import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict


class DamageNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        encoder_name = config['model']['name']
        encoder_weights = config['model']['encoder_weights']
        attention_type = config['model']['attention_type']
        num_classes = config['stage2']['num_classes']

        # Shared encoder; called twice, once for pre and once for post
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        encoder_channels = self.encoder.out_channels

        decoder_channels = (256, 128, 64, 32, 16)

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=[c * 3 for c in encoder_channels],
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=attention_type,
        )

        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        features_pre = self.encoder(pre)
        features_post = self.encoder(post)

        features_diff = [p - r for p, r in zip(features_post, features_pre)]

        features_combined = [
            torch.cat([fp, fr, fd], dim=1)
            for fp, fr, fd in zip(features_pre, features_post, features_diff)
        ]

        decoder_output = self.decoder(*features_combined)
        return self.head(decoder_output)


class LocalizationNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=config['model']['name'],
            encoder_weights=config['model']['encoder_weights'],
            in_channels=config['stage1']['in_channels'],
            classes=config['stage1']['num_classes'],
            activation=None,
            decoder_attention_type=config['model']['attention_type'],
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)