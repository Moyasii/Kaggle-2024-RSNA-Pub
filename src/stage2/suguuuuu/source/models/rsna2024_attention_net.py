from typing import Optional

import numpy as np

import torch
import torch.nn as nn

import timm

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.models.utils import GeM, FocalLoss, RSNA2024Loss  # noqa


class RSNA2024TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Self-attention block
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class RSNA2024AttentionNet(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 num_degenerative_conditions: int = 5,
                 num_segments: int = 5,
                 num_classes: int = 3,
                 use_planes: list = ['sagittal_t1', 'sagittal_t2', 'axial_t2'],
                 ):
        super().__init__()
        self._num_degenerative_conditions = num_degenerative_conditions
        self._num_segments = num_segments
        self._num_classes = num_classes
        self._use_planes = use_planes

        self._in_channels = timm_model.get('in_chans', 3)

        base_model = None
        for plane in self._use_planes:
            base_model = timm.create_model(**timm_model)
            layers = list(base_model.children())[:-2]
            setattr(self, f'{plane}_backbone', nn.Sequential(*layers))
            setattr(self, f'{plane}_gap', GeM(flatten=True, p_trainable=True))

        self.backbone = nn.Sequential(*layers)
        if "efficientnet" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif "swin" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif timm_model['model_name'].startswith('convnext_'):
            backbone_out_channels = base_model.head.in_features
        elif hasattr(timm_model['model_name'], "fc"):
            backbone_out_channels = base_model.fc.in_features
        else:
            raise ValueError(f'{timm_model["model_name"]} is not supported.')

        # Attentionの設定
        self.transformer = RSNA2024TransformerBlock(input_dim=backbone_out_channels, num_heads=8, ff_hidden_dim=512)

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.fc = nn.Linear(backbone_out_channels, num_degenerative_conditions * num_segments * num_classes)
        self.level_fc = nn.Linear(backbone_out_channels, num_segments * 5)

        self.target_loss = self._build_loss(**loss)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_loss(self,
                    name: str,
                    **kwargs: dict) -> nn.Module:
        if name == 'RSNA2024Loss':
            loss = RSNA2024Loss(**kwargs)
        else:
            raise ValueError(f'{name} is not supported.')

        return loss

    def forward(self,
                sagittal_t1_images: torch.Tensor,
                sagittal_t2_images: torch.Tensor,
                axial_t2_images: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                level_targets: Optional[torch.Tensor] = None,
                force_loss_execute: bool = False,
                ) -> dict:
        outputs = dict()

        images = dict(
            sagittal_t1=sagittal_t1_images,
            sagittal_t2=sagittal_t2_images,
            axial_t2=axial_t2_images,
        )

        feats = dict()
        for plane in self._use_planes:
            plane_images = images[plane]
            plane_backbone = getattr(self, f'{plane}_backbone')
            plane_gap = getattr(self, f'{plane}_gap')

            batch_size, plane_num_slices, h, w = plane_images.shape
            plane_images = plane_images.view(-1, self._in_channels, h, w)  # (batch_size * plane_num_slices, 1, h, w)
            plane_feats = plane_backbone(plane_images)  # (batch_size * plane_num_slices, feature_dim, h / 32, w / 32)
            plane_feats = plane_gap(plane_feats)  # (batch_size * plane_num_slices, feature_dim)
            plane_feats = plane_feats.view(batch_size, plane_num_slices, -1)  # (batch_size, plane_num_slices, feature_dim)

            feats[plane] = plane_feats

        # 特徴量を結合して1つのシーケンスに統合
        combined_features = torch.cat(list(feats.values()), dim=1)  # (batch_size, num_slices, feature_dim)

        # Transformerを適用
        combined_features = combined_features.permute(1, 0, 2)  # (num_slices, batch_size, feature_dim)
        combined_features = self.transformer(combined_features)  # (num_slices, batch_size, feature_dim)
        combined_features = combined_features.mean(dim=0)  # (batch_size, feature_dim)

        if self.training:
            logits = sum([self.fc(dropout(combined_features)) for dropout in self.dropouts]) / len(self.dropouts)
            level_logits = sum([self.level_fc(dropout(combined_features)) for dropout in self.dropouts]) / len(self.dropouts)
        else:
            logits = self.fc(combined_features)
            level_logits = self.level_fc(combined_features)

        logits = logits.view(-1, self._num_degenerative_conditions, self._num_segments, self._num_classes)
        level_logits = level_logits.view(-1, self._num_segments, 5)

        outputs['logits'] = logits
        outputs['level_logits'] = level_logits

        # model.train() で self.training == True になる
        if self.training or force_loss_execute:
            losses = self.loss(logits, targets, level_logits, level_targets)
            outputs['losses'] = losses

        return outputs

    def loss(self,
             logits: torch.Tensor,
             targets: torch.Tensor,
             level_logits: Optional[torch.Tensor] = None,
             level_targets: Optional[torch.Tensor] = None,
             ) -> dict:
        """Loss算出

        logits: (batch_size, num_degenerative_conditions, num_segments, num_classes)
        targets: (batch_size, num_degenerative_conditions, num_segments)
        """
        return self.target_loss(logits, targets, level_logits, level_targets)


if __name__ == '__main__':
    import torch.nn.functional as F  # noqa
    model = RSNA2024AttentionNet(timm_model={"model_name": "efficientnet_b0", "pretrained": False,
                                             'features_only': False, 'in_chans': 1, 'drop_rate': 0.3, 'drop_path_rate': 0.2},
                                 loss={"name": "RSNA2024Loss", "level_loss_weight": 0.1},
                                 num_degenerative_conditions=5,
                                 num_segments=5,
                                 num_classes=3,)
    model(torch.randn(4, 15, 224, 224),
          torch.randn(4, 15, 224, 224),
          torch.randn(4, 10, 224, 224),
          F.one_hot(torch.randint(0, 3, (4, 5, 5)), num_classes=3).float(),
          torch.randint(0, 5, (4, 5)))
