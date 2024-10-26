# %% [markdown]
# # Import Libralies

# %%
from typing import Optional

import os
import re
import glob
import math
import shutil
import argparse
from pathlib import Path

import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from omegaconf import OmegaConf

from src.utils import load_settings

# %% [markdown]
# # Configs

# %%
parser = argparse.ArgumentParser()
parser.add_argument('user', type=str)
parser.add_argument('seed', type=int)
args = parser.parse_args()
SETTINGS = load_settings()
ROOT_DIR = SETTINGS.raw_data_dir / 'rsna-2024-lumbar-spine-degenerative-classification'
LUMBER_COOD_DATASET_ROOT = SETTINGS.raw_data_dir / 'lumbar_coordinate_dataset'
DST_DIR = SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v4'
DST_DIR.mkdir(exist_ok=True, parents=True)

NUM_FOLDS = 5
SEED = args.seed

NG_LABELS = [
    {'study_id': 38281420, 'series_id': 880361156, 'instance_number': 9},
    # {'study_id': 286903519, 'series_id': 1921917205, 'instance_number': 13},  # 5点存在しないので除外
    # {'study_id': 665627263, 'series_id': 2231471633, 'instance_number': 9},  # 5点存在しないので除外
    {'study_id': 1438760543, 'series_id': 737753815, 'instance_number': 9},
    {'study_id': 1510451897, 'series_id': 1488857550, 'instance_number': 9},
    {'study_id': 1880970480, 'series_id': 3736941525, 'instance_number': 8},
    # {'study_id': 1901348744, 'series_id': 1490272456, 'instance_number': 11},  # 5点存在しないので除外
    {'study_id': 2151467507, 'series_id': 3086719329, 'instance_number': 8},
    # {'study_id': 2316015842, 'series_id': 1485193299, 'instance_number': 13},  # 5点存在しないので除外
    {'study_id': 2444340715, 'series_id': 3521409198, 'instance_number': 10},
    {'study_id': 2905025904, 'series_id': 816381378, 'instance_number': 11},
    {'study_id': 372642770, 'series_id': 1416772288, 'instance_number': 8},
    {'study_id': 2480600394, 'series_id': 2969009171, 'instance_number': 12},
    {'study_id': 1820866003, 'series_id': 131094096, 'instance_number': 7},
    {'study_id': 2151509334, 'series_id': 3072714332, 'instance_number': 9},
    {'study_id': 3819260179, 'series_id': 1735851779, 'instance_number': 7},
]

# KEYPOIINT MODEL
KEYPOINT_MODEL_INFO = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v2_0008_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v2/0008/exp0008_fold0_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v2_0008_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v2/0008/exp0008_fold1_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v2_0008_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v2/0008/exp0008_fold2_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v2_0008_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v2/0008/exp0008_fold3_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v2_0008_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v2/0008/exp0008_fold4_best_score.pth',
    },
]
KEYPOINT_FIND_METHOD = 'percentile'


CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
SEGMENTS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
PLANES = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']
DEVICE = 0

# %% [markdown]
# ## Helper functions

# %%


def glob_sorted(p: str) -> list[str]:
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    files = glob.glob(p)
    files = sorted(files, key=natural_keys)
    return files

# %%


def apply_min_max_norm(x, lower=0.1, upper=99.9):  # 1, 99 #0.05, 99.5 #0, 100
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

# %%


def convert_image_to_world_coord(
    image_position: list,
    image_orientation: list,
    pixel_spacing: list,
    i: int,
    j: int,
) -> tuple[float, float, float]:
    sx = image_position[0]
    sy = image_position[1]
    sz = image_position[2]
    xx_di = image_orientation[0] * pixel_spacing[0]
    xy_di = image_orientation[1] * pixel_spacing[0]
    xz_di = image_orientation[2] * pixel_spacing[0]
    yx_dj = image_orientation[3] * pixel_spacing[1]
    yy_dj = image_orientation[4] * pixel_spacing[1]
    yz_dj = image_orientation[5] * pixel_spacing[1]
    px = (xx_di * i) + (yx_dj * j) + sx
    py = (xy_di * i) + (yy_dj * j) + sy
    pz = (xz_di * i) + (yz_dj * j) + sz
    return px, py, pz


# %%
def convert_image_to_world_coord_array(
    image_position: np.ndarray,
    image_orientation: np.ndarray,
    pixel_spacing: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
) -> tuple[float, float, float]:
    sx = image_position[..., 0]
    sy = image_position[..., 1]
    sz = image_position[..., 2]
    xx_di = image_orientation[..., 0] * pixel_spacing[..., 0]
    xy_di = image_orientation[..., 1] * pixel_spacing[..., 0]
    xz_di = image_orientation[..., 2] * pixel_spacing[..., 0]
    yx_dj = image_orientation[..., 3] * pixel_spacing[..., 1]
    yy_dj = image_orientation[..., 4] * pixel_spacing[..., 1]
    yz_dj = image_orientation[..., 5] * pixel_spacing[..., 1]
    px = (xx_di * i) + (yx_dj * j) + sx
    py = (xy_di * i) + (yy_dj * j) + sy
    pz = (xz_di * i) + (yz_dj * j) + sz
    return np.asarray([px, py, pz]).T

# %%


def load_dicom_stack(dicom_folder: str, plane: str) -> dict:
    # https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes
    dicom_files = glob_sorted(os.path.join(dicom_folder, "*.dcm"))
    dicom_files = np.asarray(dicom_files)

    arrays = []
    image_shapes = []
    image_position_patients = []
    image_orientation_patients = []
    pixel_spacings = []
    instance_numbers = []
    for f in dicom_files:
        d = pydicom.dcmread(f)
        arrays.append(d.pixel_array.astype("float32"))
        image_shapes.append(d.pixel_array.shape)
        image_position_patients.append(d.ImagePositionPatient)
        image_orientation_patients.append(d.ImageOrientationPatient)
        pixel_spacings.append(d.PixelSpacing)
        instance_numbers.append(d.InstanceNumber)

    image_shapes = np.asarray(image_shapes)
    image_position_patients = np.asarray(image_position_patients).astype("float")
    image_orientation_patients = np.asarray(image_orientation_patients).astype("float")
    pixel_spacings = np.asarray(pixel_spacings).astype("float")
    instance_numbers = np.asarray(instance_numbers).astype("int")

    plane_index = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    origin_world_positions = []
    standard_positions = []
    for ipp, iop, ps in zip(image_position_patients, image_orientation_patients, pixel_spacings):
        origin_world_position = convert_image_to_world_coord(ipp, iop, ps, 0, 0)
        origin_world_positions.append(origin_world_position)
        standard_position = origin_world_position[plane_index]
        standard_positions.append(standard_position)

    origin_world_positions = np.asarray(origin_world_positions)
    standard_positions = np.asarray(standard_positions)

    if plane.lower() == "sagittal":
        sorted_indices = np.argsort(standard_positions)
    else:
        sorted_indices = np.argsort(-standard_positions)
    return {
        "dicom_files": dicom_files[sorted_indices],
        "instance_numbers": instance_numbers[sorted_indices],
        # "arrays": arrays[sorted_indices],
        "arrays": [arrays[i] for i in sorted_indices],
        "image_shapes": image_shapes[sorted_indices],
        "image_position_patients": image_position_patients[sorted_indices],
        "image_orientation_patients": image_orientation_patients[sorted_indices],
        "pixel_spacings": pixel_spacings[sorted_indices],
        "origin_world_positions": origin_world_positions[sorted_indices],
        "standard_positions": standard_positions[sorted_indices],
    }

# %%


def get_config(config_path: str, dot_list: list) -> dict:
    config_omega_from_yaml = OmegaConf.load(config_path)
    config_omega_from_args = OmegaConf.from_dotlist(dot_list)
    config_omega = OmegaConf.merge(config_omega_from_yaml, config_omega_from_args)
    config = OmegaConf.to_container(config_omega, resolve=True)  # DictConfig -> dict
    return config

# %%


def draw_keypoints(image, annotation, color=(255, 0, 0), radius=3, levels=SEGMENTS):
    level_colors = {
        'L1/L2': (255, 0, 0),
        'L2/L3': (0, 255, 0),
        'L3/L4': (0, 0, 255),
        'L4/L5': (255, 191, 0),
        'L5/S1': (0, 255, 255),
    }

    image = image.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for level in levels:
        x, y = annotation[f'{level}_x'], annotation[f'{level}_y']
        radius = image.shape[1] // 50
        cv2.circle(image, (int(x), int(y)), radius, level_colors[level], -1)
        cv2.putText(image, f'{level} ({x:.1f}, {y:.1f})', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, level_colors[level], 1, cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# %% [markdown]
# # Reading and Taking a look csv


# %%
DF = pd.read_csv(f'{ROOT_DIR}/train_series_descriptions.csv')
DF.head()

# %%
COOD_DF = pd.read_csv(f'{ROOT_DIR}/train_label_coordinates.csv')
COOD_DF.head()

# %%
LUMBER_DATASET_COOD_DF = pd.read_csv(f'{LUMBER_COOD_DATASET_ROOT}/coords_pretrain.csv')
LUMBER_DATASET_COOD_DF


# %% [markdown]
# ## NGラベルを手修正する

# %%
fixed_coods = [
    [{'study_id': 38281420, 'series_id': 880361156, 'instance_number': 9, 'level': 'L5/S1', 'x': 188, 'y': 215}],
    [{'study_id': 1438760543, 'series_id': 737753815, 'instance_number': 9, 'level': 'L1/L2', 'x': 237, 'y': 150}],
    [{'study_id': 1510451897, 'series_id': 1488857550, 'instance_number': 9, 'level': 'L4/L5', 'x': 174, 'y': 200}],
    [{'study_id': 1880970480, 'series_id': 3736941525, 'instance_number': 8, 'level': 'L5/S1', 'x': 265, 'y': 410}],
    [{'study_id': 2151467507, 'series_id': 3086719329, 'instance_number': 8, 'level': 'L3/L4', 'x': 290, 'y': 263}],
    [{'study_id': 2444340715, 'series_id': 3521409198, 'instance_number': 10, 'level': 'L5/S1', 'x': 205, 'y': 307}],
    [{'study_id': 2905025904, 'series_id': 816381378, 'instance_number': 11, 'level': 'L1/L2', 'x': 198, 'y': 130}],
    [{'study_id': 372642770, 'series_id': 1416772288, 'instance_number': 8, 'level': 'L1/L2', 'x': 287, 'y': 182}],
    [{'study_id': 2480600394, 'series_id': 2969009171, 'instance_number': 12, 'level': 'L2/L3', 'x': 190, 'y': 150}],
    [{'study_id': 1820866003, 'series_id': 131094096, 'instance_number': 7, 'level': 'L2/L3', 'x': 280, 'y': 180}],
    [{'study_id': 2151509334, 'series_id': 3072714332, 'instance_number': 9, 'level': 'L4/L5', 'x': 290, 'y': 300}],
    [{'study_id': 3819260179, 'series_id': 1735851779, 'instance_number': 7, 'level': 'L5/S1', 'x': 265, 'y': 340}],
]

FIXED_COOD_DF = COOD_DF.copy()
for fixed_cood in fixed_coods:
    fixed_cood_df = pd.DataFrame(fixed_cood)
    for i, row in fixed_cood_df.iterrows():
        FIXED_COOD_DF.loc[(FIXED_COOD_DF['study_id'] == row['study_id']) &
                          (FIXED_COOD_DF['series_id'] == row['series_id']) &
                          (FIXED_COOD_DF['instance_number'] == row['instance_number']) &
                          (FIXED_COOD_DF['level'] == row['level']), ['x', 'y']] = row[['x', 'y']].values
FIXED_COOD_DF.to_csv(DST_DIR / 'train_label_coordinates_fixed.csv', index=False)

# %% [markdown]
# ## RSNA2024のキーポイントデータセットを作成
#
# RSNA2024のtrain_label_coordinates.csvをベースにキーポイントのデータセットを作成する。

# %%
if not (DST_DIR / 'train_gt.csv').exists():
    data = []
    st_ids = DF['study_id'].unique()
    descs = ['Sagittal T1', 'Sagittal T2/STIR']

    for idx, study_id in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = DF[DF['study_id'] == study_id]

        for desc in descs:
            for series_id in pdf[pdf['series_description'] == desc]['series_id'].values.tolist():
                # X軸の中心だと脊柱管狭窄が検出しやすいため脊柱管狭窄のみを対象とする
                cood_df = FIXED_COOD_DF[(FIXED_COOD_DF['study_id'] == study_id) &
                                        (FIXED_COOD_DF['series_id'] == series_id) &
                                        (FIXED_COOD_DF['condition'] == 'Spinal Canal Stenosis')].copy()
                if len(cood_df) == 0:
                    continue
                dicom_folder = f'{ROOT_DIR}/train_images/{study_id}/{series_id}'
                stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])
                num_slices = len(stack['instance_numbers'])

                lower_idx = math.floor(num_slices * 0.35)
                upper_idx = math.ceil(num_slices * 0.65)
                target_instances = stack['instance_numbers'][lower_idx:upper_idx]
                cood_df = cood_df[cood_df['instance_number'].isin(target_instances)]
                count_df = cood_df[['instance_number', 'condition']].value_counts().reset_index()
                count_df = count_df[count_df['count'] == 5]
                if len(count_df) == 0:
                    # 座標が5点存在しない場合はスキップ
                    continue

                # X軸昇順（体の右から左にかけて）にスライス画像を保存する
                image_dir = DST_DIR / f'train_images_png/{study_id}/{series_id}'
                image_dir.mkdir(parents=True, exist_ok=True)
                for stack_idx in range(num_slices):
                    image = apply_min_max_norm(stack['arrays'][stack_idx])
                    image_path = image_dir / f'{stack_idx:04d}_{stack["instance_numbers"][stack_idx]:04d}_{stack["standard_positions"][stack_idx]:.2f}.png'
                    cv2.imwrite(image_path.as_posix(), image)

                for i, count_row in count_df.iterrows():
                    instance_number = count_row.instance_number
                    condition = count_row.condition

                    stack_idx = np.where(stack['instance_numbers'] == instance_number)[0][0]
                    standard_pos = stack['standard_positions'][stack_idx]
                    image_shape = stack['image_shapes'][stack_idx]
                    anno = dict(
                        source='rsna',
                        study_id=study_id,
                        series_id=series_id,
                        desc=desc,
                        instance_number=instance_number,
                        condition=condition,
                        standard_pos=standard_pos,
                        image_dir=image_dir.relative_to(DST_DIR),
                        height=image_shape[0],
                        width=image_shape[1],
                    )

                    instance_cood_df = cood_df[cood_df['instance_number'] == instance_number].copy()

                    # 座標を追加
                    for i, cood_row in instance_cood_df.iterrows():
                        segment = cood_row.level
                        anno[segment + '_x'] = cood_row.x
                        anno[segment + '_y'] = cood_row.y
                    # 正規化座標を追加
                    for i, cood_row in instance_cood_df.iterrows():
                        segment = cood_row.level
                        anno[segment + '_nx'] = cood_row.x / image_shape[1]
                        anno[segment + '_ny'] = cood_row.y / image_shape[0]

                    data.append(anno)

    train_gt_df = pd.DataFrame(data=data)
    train_gt_df = train_gt_df.drop_duplicates()
    train_gt_df.to_csv(DST_DIR / 'train_gt.csv', index=False)


# %% [markdown]
# ## RSNAのラベルなしデータに疑似ラベルを付ける
#
# Ver2の学習済みモデルを使ってラベルなしデータに疑似ラベルを付ける。

# %% [markdown]
# ### キーポイントモデル

# %%
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


class RSNA2024KeypointNetTest(nn.Module):
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

    @ property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self,
                images: torch.Tensor,
                ) -> dict:
        outputs = dict()

        feats = self.backbone(images)
        feats = self.neck(feats)
        logits = self.head(feats)
        outputs['logits'] = logits
        return outputs

# %%


class RSNA2024KeypointDetector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 model_class=RSNA2024KeypointNetTest,
                 threshold: float = 0.0,
                 keypoint_find_method: str = 'max',
                 segments: list = SEGMENTS,
                 condition: list = CONDITIONS,
                 planes: list = PLANES):
        self._model_class = model_class
        self._model_infos = model_infos
        self._device = device
        if keypoint_find_method == 'max':
            self._find_keypoint = self._find_keypoint_by_max
        elif keypoint_find_method == 'percentile':
            self._find_keypoint = self._find_keypoint_by_percentile
        else:
            raise ValueError(f'find_keypoint: {keypoint_find_method} is not supported.')
        self._threshold = threshold
        self._segments = segments
        self._condition = condition
        self._planes = planes
        self._models = []
        self._transforms = []

        for model_info in model_infos:
            config_path = model_info['config']
            checkpoint = model_info['checkpoint']

            config = get_config(config_path, [])
            config['model']['timm_model']['pretrained'] = False
            model = model_class(**config['model'])
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            model.to(device)
            self._models.append(model)
            self._transforms.append(A.Compose([
                A.Resize(*config['transform']['image_size']),
                A.Normalize(mean=0.5, std=0.5)
            ]))

    def _preprocess(self, image: np.ndarray, transform: A.Compose, device: torch.device) -> torch.Tensor:
        transformed_image = transform(image=image)['image']
        transformed_image = torch.Tensor(transformed_image[None, None, ...]).to(device)
        return transformed_image

    def _find_keypoint_by_max(self, heatmap: np.ndarray, threshold: float) -> np.ndarray:
        heatmap[heatmap < threshold] = 0
        p_y, p_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        p_y, p_x = (p_y + 0.5), (p_x + 0.5)
        return p_x, p_y

    def _find_keypoint_by_percentile(self, heatmap: np.ndarray, threshold: float) -> np.ndarray:
        heatmap[heatmap < threshold] = 0
        percentile = np.percentile(heatmap, 99.8)
        mask = heatmap >= percentile
        mask_indexes = np.where(mask)
        y_min = np.min(mask_indexes[0]) + 0.5
        y_max = np.max(mask_indexes[0]) + 0.5
        x_min = np.min(mask_indexes[1]) + 0.5
        x_max = np.max(mask_indexes[1]) + 0.5
        p_x, p_y = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        return p_x, p_y

    def _postprocess(self,
                     heatmap: np.ndarray,
                     image_shape: list,
                     image_position_patient: list,
                     image_orientation_patient: list,
                     pixel_spacings: list,
                     keypoint_threshold: float) -> tuple:
        segment_image_positions = {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None}
        segment_world_positions = {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None}

        hmap_height, hmap_width = heatmap.shape[-2:]
        for i, segment in enumerate(segment_image_positions.keys()):
            hmap_x, hmap_y = self._find_keypoint(heatmap[0][i], keypoint_threshold)
            norm_x, norm_y = (hmap_x / hmap_width, hmap_y / hmap_height)
            x, y = (norm_x * image_shape[1], norm_y * image_shape[0])
            segment_image_positions[segment] = (norm_x, norm_y)
            segment_world_positions[segment] = convert_image_to_world_coord(image_position_patient,
                                                                            image_orientation_patient,
                                                                            pixel_spacings,
                                                                            x, y)

        return segment_image_positions, segment_world_positions

    def predict(
        self,
        stack: Optional[dict],
    ) -> dict[str, list[float]]:
        if stack is None:
            return ({"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None},
                    {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None})
        num_slices = len(stack['instance_numbers'])
        stack_index = num_slices // 2
        image = apply_min_max_norm(stack['arrays'][stack_index])

        heatmap_list = []
        for model, transform in zip(self._models, self._transforms):
            processed_image = self._preprocess(image, transform, self._device)
            with torch.no_grad():
                logit = model(processed_image)['logits']
            heatmap = logit.sigmoid().detach().cpu().numpy()
            heatmap_list.append(heatmap)
        heatmap = np.concatenate(heatmap_list, axis=0).mean(axis=0, keepdims=True)

        image_shape = stack['image_shapes'][stack_index]
        image_position_patient = stack['image_position_patients'][stack_index]
        image_orientation_patient = stack['image_orientation_patients'][stack_index]
        pixel_spacings = stack['pixel_spacings'][stack_index]
        segment_image_positions, segment_world_positions = self._postprocess(heatmap, image_shape,
                                                                             image_position_patient, image_orientation_patient,
                                                                             pixel_spacings, self._threshold)

        return segment_image_positions, segment_world_positions


# %%
if not (DST_DIR / 'train_pseudo.csv').exists():
    data = []
    keypoint_detector = RSNA2024KeypointDetector(model_infos=KEYPOINT_MODEL_INFO, device=DEVICE, keypoint_find_method=KEYPOINT_FIND_METHOD)
    train_gt_df = pd.read_csv(DST_DIR / 'train_gt.csv')
    st_ids = list(set(DF['study_id'].unique()) - set(train_gt_df['study_id'].unique()))
    descs = ['Sagittal T2/STIR', 'Sagittal T1']

    for idx, study_id in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = DF[DF['study_id'] == study_id]

        for desc in descs:
            series_ids = pdf[pdf['series_description'] == desc]['series_id'].values.tolist()
            if len(series_ids) == 0:
                # T2画像がない場合はT1画像を使用する
                continue

            series_id = series_ids[0]
            dicom_folder = f'{ROOT_DIR}/train_images/{study_id}/{series_id}'
            stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])

            # 椎間板レベルの位置を予測
            coords, _ = keypoint_detector.predict(stack)

            num_slices = len(stack['instance_numbers'])

            # X軸昇順（体の右から左にかけて）にスライス画像を保存する
            image_dir = DST_DIR / f'train_images_png/{study_id}/{series_id}'
            image_dir.mkdir(parents=True, exist_ok=True)
            for stack_idx in range(num_slices):
                image = apply_min_max_norm(stack['arrays'][stack_idx])
                image_path = image_dir / f'{stack_idx:04d}_{stack["instance_numbers"][stack_idx]:04d}_{stack["standard_positions"][stack_idx]:.2f}.png'
                cv2.imwrite(image_path.as_posix(), image)

            stack_idx = num_slices // 2
            instance_number = stack['instance_numbers'][stack_idx]
            standard_pos = stack['standard_positions'][stack_idx]
            image_shape = stack['image_shapes'][stack_idx]

            anno = dict(
                source='rsna',
                study_id=study_id,
                series_id=series_id,
                desc=desc,
                instance_number=instance_number,
                condition='',
                standard_pos=standard_pos,
                image_dir=image_dir.relative_to(DST_DIR),
                height=image_shape[0],
                width=image_shape[1],
            )

            # 座標を追加
            for level in SEGMENTS:
                anno[level + '_x'] = coords[level][0] * image_shape[1]
                anno[level + '_y'] = coords[level][1] * image_shape[0]
            # 正規化座標を追加
            for level in SEGMENTS:
                anno[level + '_nx'] = coords[level][0]
                anno[level + '_ny'] = coords[level][1]

            data.append(anno)

            # Sagittal T2/STIRに疑似ラベルがつけられたら次のStudyに移る
            # もしT2付けられなかったらT1に疑似ラベルをつける
            break

    train_pseudo_df = pd.DataFrame(data=data)
    train_pseudo_df = train_pseudo_df.drop_duplicates()
    train_pseudo_df.to_csv(DST_DIR / 'train_pseudo.csv', index=False)

# %% [markdown]
# ## 誤検出しているラベルを手修正
#
# 500件の疑似ラベルデータを目視した。ミスっているアノテをここで修正する。

# %%
train_pseudo_df = pd.read_csv(DST_DIR / 'train_pseudo.csv')
pseudo_fix_data = [
    # 位置ずれ
    [404602713, 1230697721, 11, {
        'L1/L2_x': 153.757131301089, 'L1/L2_y': 118.448895446611,
        'L2/L3_x': 149.397458276566, 'L2/L3_y': 151.548350487482,
        'L3/L4_x': 152.885196696185, 'L3/L4_y': 187.331729234076,
        'L4/L5_x': 160.732608140326, 'L4/L5_y': 213.489767381215,
        'L5/S1_x': 177.299365633514, 'L5/S1_y': 230.908030164339,
    }],
    # 位置ずれ
    [953639220, 270540558, 10, {
        'L1/L2_x': 266.756671899529, 'L1/L2_y': 174.015698587127,
        'L2/L3_x': 271.579277864992, 'L2/L3_y': 217.419152276295,
        'L3/L4_x': 279.616954474097, 'L3/L4_y': 251.177394034536,
        'L4/L5_x': 278.009419152276, 'L4/L5_y': 294.580847723704,
        'L5/S1_x': 270.775510204081, 'L5/S1_y': 339.591836734693,
    }],
    # 位置ずれ
    [1088270559, 648725109, 11, {
        'L1/L2_x': 241.5157894736842, 'L1/L2_y': 149.55789473684212,
        'L2/L3_x': 242.52631578947367, 'L2/L3_y': 199.0736842105263,
        'L3/L4_x': 234.44210526315788, 'L3/L4_y': 248.63157894736844,
        'L4/L5_x': 233.4315789473684, 'L4/L5_y': 286.9894736842105,
        'L5/S1_x': 242.52631578947367, 'L5/S1_y': 320.9894736842105,
    }],
    # 位置ずれ
    [1314603564, 1610188106, 9, {
        'L1/L2_x': 211.0, 'L1/L2_y': 101.0,
        'L2/L3_x': 197.0, 'L2/L3_y': 147.0,
        'L3/L4_x': 188.0, 'L3/L4_y': 186.0,
        'L4/L5_x': 188.0, 'L4/L5_y': 222.0,
        'L5/S1_x': 206.0, 'L5/S1_y': 259.0,
    }],
    # 左右湾曲
    [1647904243, 429524619, 14, {
        'L1/L2_x': 167.092105263157, 'L1/L2_y': 55.438596491228,
        'L2/L3_x': 158.109649122807, 'L2/L3_y': 83.5087719298245,
        'L3/L4_x': 145.197368421052, 'L3/L4_y': 111.017543859649,
        'L4/L5_x': 137.337719298245, 'L4/L5_y': 137.964912280701,
        'L5/S1_x': 142.951754385964, 'L5/S1_y': 168.842105263157,
    }],
    # 左右湾曲
    [1755159626, 2513905084, 11, {
        'L1/L2_x': 302.997821350762, 'L1/L2_y': 165.647058823529,
        'L2/L3_x': 295.18954248366, 'L2/L3_y': 213.612200435729,
        'L3/L4_x': 284.034858387799, 'L3/L4_y': 261.57734204793,
        'L4/L5_x': 294.074074074074, 'L4/L5_y': 309.54248366013,
        'L5/S1_x': 311.921568627451, 'L5/S1_y': 356.392156862745,
    }],
    # 位置ずれ
    [2466046363, 1478301987, 6, {
        'L1/L2_x': 277.051039697542, 'L1/L2_y': 191.879017013232,
        'L2/L3_x': 274.147448015122, 'L2/L3_y': 239.304347826086,
        'L3/L4_x': 287.697542533081, 'L3/L4_y': 284.793950850661,
        'L4/L5_x': 312.862003780718, 'L4/L5_y': 320.604914933837,
        'L5/S1_x': 330.283553875236, 'L5/S1_y': 344.801512287334,
    }],
    # 位置ずれ
    [2918363467, 4185472245, 12, {
        'L1/L2_x': 165.47195622435, 'L1/L2_y': 133.953488372093,
        'L2/L3_x': 160.218878248974, 'L2/L3_y': 165.909712722298,
        'L3/L4_x': 159.343365253078, 'L3/L4_y': 201.805745554035,
        'L4/L5_x': 171.600547195622, 'L4/L5_y': 221.504787961696,
        'L5/S1_x': 188.235294117647, 'L5/S1_y': 235.95075239398,
    }],
    # 位置ずれ
    [3284652867, 475861931, 11, {
        'L1/L2_x': 272.0, 'L1/L2_y': 47.0,
        'L2/L3_x': 264.579710144927, 'L2/L3_y': 108.95652173913,
        'L3/L4_x': 257.27536231884, 'L3/L4_y': 170.63768115942,
        'L4/L5_x': 257.27536231884, 'L4/L5_y': 233.130434782608,
        'L5/S1_x': 262.95652173913, 'L5/S1_y': 289.130434782608,
    }],
    # 左右湾曲
    [3781188430, 2968472419, 10, {
        'L1/L2_x': 315.992805936995, 'L1/L2_y': 187.19224555735,
        'L2/L3_x': 300.277135500807, 'L2/L3_y': 244.264943457189,
        'L3/L4_x': 272.154356825525, 'L3/L4_y': 284.794830371567,
        'L4/L5_x': 257.26582693861, 'L4/L5_y': 333.596122778675,
        'L5/S1_x': 280.425762318255, 'L5/S1_y': 371.644588045234,
    }],
    # 左右湾曲
    [3828017267, 1300874959, 8, {
        'L1/L2_x': 213.565217391304, 'L1/L2_y': 121.217391304347,
        'L2/L3_x': 211.478260869565, 'L2/L3_y': 162.95652173913,
        'L3/L4_x': 204.521739130434, 'L3/L4_y': 198.434782608695,
        'L4/L5_x': 197.565217391304, 'L4/L5_y': 236.695652173913,
        'L5/S1_x': 201.739130434782, 'L5/S1_y': 260.347826086956,
    }],]

n_cols = 2
n_rows = len(pseudo_fix_data)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), tight_layout=True)
for i, (study_id, series_id, instance_number, fix_kpt) in enumerate(pseudo_fix_data):
    ng_anno = train_pseudo_df[(train_pseudo_df['study_id'] == study_id) &
                              (train_pseudo_df['series_id'] == series_id) &
                              (train_pseudo_df['instance_number'] == instance_number)].iloc[0].to_dict()
    # 修正版座標を正規化
    for level in SEGMENTS:
        fix_kpt[f'{level}_nx'] = fix_kpt[f'{level}_x'] / ng_anno['width']
        fix_kpt[f'{level}_ny'] = fix_kpt[f'{level}_y'] / ng_anno['height']

    # アノテーション情報を修正
    train_pseudo_df.loc[(train_pseudo_df['study_id'] == study_id) &
                        (train_pseudo_df['series_id'] == series_id) &
                        (train_pseudo_df['instance_number'] == instance_number),
                        list(fix_kpt.keys())] = list(fix_kpt.values())

    fixed_anno = train_pseudo_df[(train_pseudo_df['study_id'] == study_id) &
                                 (train_pseudo_df['series_id'] == series_id) &
                                 (train_pseudo_df['instance_number'] == instance_number)].iloc[0].to_dict()

    image_dir = DST_DIR / f'train_images_png/{study_id}/{series_id}'
    image_path_list = sorted(image_dir.iterdir())
    instance_number_to_image_path = {int(p.stem.split('_')[1]): p for p in image_path_list}
    image = cv2.imread(instance_number_to_image_path[instance_number])
    colors = {
        'L1/L2': (1.0, 0.0, 0.0),
        'L2/L3': (0.0, 1.0, 0.0),
        'L3/L4': (0.0, 0.0, 1.0),
        'L4/L5': (1.0, 0.75, 0.0),
        'L5/S1': (0.0, 1.0, 1.0),
    }
    axes[i][0].imshow(image)
    axes[i][0].set_title(f'{study_id}_{series_id}_{instance_number} (ng)')
    axes[i][1].imshow(image)
    axes[i][1].set_title(f'{study_id}_{series_id}_{instance_number} (fixed)')
    for level in SEGMENTS:
        nx, ny = ng_anno[f'{level}_nx'], ng_anno[f'{level}_ny']
        x, y = nx * ng_anno['width'], ny * ng_anno['height']
        axes[i][0].plot(x, y, 'o', color=colors[level])
        axes[i][0].text(x + 10, y, f"{level} ({x:.2f},{y:.2f})", color=colors[level], fontsize=10)
        nx, ny = fixed_anno[f'{level}_nx'], fixed_anno[f'{level}_ny']
        x, y = nx * fixed_anno['width'], ny * fixed_anno['height']
        axes[i][1].plot(x, y, 'o', color=colors[level])
        axes[i][1].text(x + 10, y, f"{level} ({x:.2f},{y:.2f})", color=colors[level], fontsize=10)

train_pseudo_df.to_csv(DST_DIR / 'train_pseudo_fixed.csv', index=False)

# %% [markdown]
# ## train.csvを作成

# %%
train_gt_df = pd.read_csv(DST_DIR / 'train_gt.csv')
train_pseudo_df = pd.read_csv(DST_DIR / 'train_pseudo_fixed.csv')
train_df = pd.concat([train_gt_df, train_pseudo_df], ignore_index=True)
train_df = train_df.sort_values(['study_id', 'series_id', 'instance_number']).reset_index(drop=True)
train_df.to_csv(DST_DIR / 'train.csv', index=False)
train_df['study_id'].nunique()

# %% [markdown]
# ## Cross-Validationを作成

# %%
train_df = pd.read_csv(DST_DIR / 'train.csv')
unique_study_ids = train_df['study_id'].unique()
np.random.shuffle(unique_study_ids)

train_df['fold'] = -1
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
study_id_to_fold = {}
for fold, (_, valid_idx) in enumerate(kf.split(unique_study_ids)):
    valid_study_ids = unique_study_ids[valid_idx]
    for valid_study_id in valid_study_ids:
        study_id_to_fold[valid_study_id] = fold

train_df['fold'] = train_df['study_id'].map(study_id_to_fold)
assert sum(train_df['fold'] == -1) == 0
train_df.to_csv(DST_DIR / 'train_5folds.csv', index=False)

# %% [markdown]
# ## Lumber Coordinate Datasetの追加

# %%
if not (DST_DIR / 'train_lumber_dataset.csv').exists():
    desc_table = {'lsd': 'Sagittal T2/STIR', 'spider': 'Sagittal T2/STIR', 'osf': 'Sagittal T1', 'tseg': 'ct'}
    filenames = LUMBER_DATASET_COOD_DF['filename'].unique()

    data = []
    for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
        cood_df = LUMBER_DATASET_COOD_DF[LUMBER_DATASET_COOD_DF['filename'] == filename].copy()
        if len(cood_df) != 5:
            print(filename)
            continue

        source = cood_df['source'].unique()[0]
        if source == 'tseg':
            # CT画像のためスキップする
            continue

        src_file_path = Path(LUMBER_COOD_DATASET_ROOT) / 'data' / f'processed_{source}_jpgs' / filename

        study_id = 10000000000 + i
        series_id = 0
        instance_number = 0
        standard_pos = 0.0

        dst_dir = DST_DIR / 'lumber_images_jpg' / f'processed_{source}_jpgs/{study_id}/{series_id}'
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
        dst_filename = f'0000_{instance_number:04d}_{standard_pos:.2f}.jpg'
        dst_file_path = dst_dir / dst_filename
        shutil.copy2(src_file_path, dst_file_path)

        image_shape = cv2.imread(dst_file_path.as_posix()).shape[:2]

        anno = dict(
            source=source,
            study_id=study_id,
            series_id=series_id,
            desc=desc_table[source],
            instance_number=instance_number,
            condition='unknown',
            standard_pos=standard_pos,
            image_dir=dst_dir.relative_to(DST_DIR),
            height=image_shape[0],
            width=image_shape[1],
        )
        # 座標を追加
        for segment in SEGMENTS:
            anno[segment + '_x'] = cood_df.loc[cood_df['level'] == segment, 'x'].values[0]
            anno[segment + '_y'] = cood_df.loc[cood_df['level'] == segment, 'y'].values[0]
        # 正規化座標を追加
        for segment in SEGMENTS:
            anno[segment + '_nx'] = cood_df.loc[cood_df['level'] == segment, 'x'].values[0] / image_shape[1]
            anno[segment + '_ny'] = cood_df.loc[cood_df['level'] == segment, 'y'].values[0] / image_shape[0]

        data.append(anno)

    train_lumber_df = pd.DataFrame(data=data)
    train_lumber_df['fold'] = -100
    random_indices = np.random.choice(train_lumber_df.index, size=30, replace=False)
    train_lumber_df.loc[random_indices, 'fold'] = -101
    train_lumber_df.to_csv(DST_DIR / 'train_lumber_dataset.csv', index=False)

# %% [markdown]
# ## 結合

# %%
train_df = pd.read_csv(DST_DIR / 'train_5folds.csv')
train_lumber_df = pd.read_csv(DST_DIR / 'train_lumber_dataset.csv')
pd.concat([train_df, train_lumber_df]).to_csv(DST_DIR / 'train_all_5folds.csv', index=False)
