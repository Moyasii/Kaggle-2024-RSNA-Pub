from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.models.utils import GeM, BFWithLogitsLoss, CenterNetFocalLoss, HeatmapSimilarityLoss  # noqa


class RSNA2024KeypointNeck(nn.Module):
    def __init__(self,
                 in_channels: list = [32, 56, 120, 208],
                 out_channel: int = 32):
        super().__init__()
        self._in_channels = in_channels
        self._out_channel = out_channel

        self.lateal_conv = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            in_c = in_channels[i]
            out_c = in_channels[i]
            self.lateal_conv.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
        self.last = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.upsample_conv = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            in_c = in_channels[i + 1]
            out_c = in_channels[i + 0]
            self.upsample_conv.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))

    def forward(self, in_feature: tuple[torch.Tensor]) -> torch.Tensor:
        x = in_feature[-1]
        for i in range(len(self._in_channels) - 1, 0, -1):
            _, h, w = in_feature[i-1].shape[1:]
            x = F.interpolate(x, size=(h, w))
            x = self.upsample_conv[i-1](x)
            lat_x = self.lateal_conv[i-1](in_feature[i-1])
            x = x + lat_x

        x = self.last(x)
        return x


class RSNA2024KeypointNet(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 neck_out_channels: int = 96,
                 head_mid_channels: int = 64,
                 num_classes: int = 5):
        super().__init__()
        self._num_classes = num_classes

        in_channles = timm_model.get('in_chans', 3)
        dummy_image = torch.randn(1, in_channles, 224, 224)

        self.backbone = timm.create_model(**timm_model)
        feats = self.backbone(dummy_image)
        backbone_out_channels = [feat.shape[1] for feat in feats]

        self.neck = RSNA2024KeypointNeck(in_channels=backbone_out_channels, out_channel=neck_out_channels)
        self.head = nn.Sequential(
            nn.Conv2d(neck_out_channels, head_mid_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.head(self.neck(self.backbone(dummy_image)))
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.target_loss = self._build_loss(**loss)

    @ property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_loss(self,
                    name: str,
                    **kwargs: dict) -> nn.Module:
        if name == 'CrossEntropyLoss':
            if 'weight' in kwargs:
                weight = kwargs.pop('weight')
                kwargs['weight'] = torch.tensor(weight)
            loss = nn.CrossEntropyLoss(**kwargs)
        elif name == 'BFWithLogitsLoss':
            loss = BFWithLogitsLoss(**kwargs)
        elif name == 'CenterNetFocalLoss':
            loss = CenterNetFocalLoss(**kwargs)
        elif name == 'HeatmapSimilarityLoss':
            loss = HeatmapSimilarityLoss(**kwargs)
        else:
            raise ValueError(f'{name} is not supported.')

        return loss

    def forward(self,
                images: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                force_loss_execute: bool = False,
                ) -> dict:
        outputs = dict()

        feats = self.backbone(images)
        feats = self.neck(feats)
        if self.training:
            logits = sum([self.head(dropout(feats)) for dropout in self.dropouts]) / len(self.dropouts)
        else:
            logits = self.head(feats)
        outputs['logits'] = logits

        # model.train() で self.training == True になる
        if self.training or force_loss_execute:
            losses = self.loss(logits, targets)
            outputs['losses'] = losses

        return outputs

    def loss(self,
             logits: torch.Tensor,
             targets: torch.Tensor,
             ) -> dict:
        losses = dict()

        loss = self.target_loss(logits.reshape(logits.shape[0], -1), targets.reshape(targets.shape[0], -1))
        losses['loss'] = loss
        return losses


if __name__ == '__main__':
    model = RSNA2024KeypointNet(timm_model={"model_name": "tf_efficientnet_b3.in1k", "in_chans": 1, "pretrained": True,
                                            "out_indices": [0, 1, 2, 3, 4], "features_only": True, },
                                loss={"name": "BFWithLogitsLoss"},
                                num_classes=5)
    model(torch.randn(4, 1, 224, 224), torch.rand(4, 5 * 112 * 112).to(torch.float32))
