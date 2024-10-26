# %% [markdown]
# # Import Libralies

# %%
from typing import Optional

import os
import re
import glob
import argparse

import pydicom
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A

from omegaconf import OmegaConf

from src.utils import load_settings

# %% [markdown]
# # Configs

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('user', type=str)
args = parser.parse_args()

SETTINGS = load_settings()
ROOT_DIR = SETTINGS.raw_data_dir / 'rsna-2024-lumbar-spine-degenerative-classification'
DST_DIR = SETTINGS.train_data_clean_dir / f'{args.user}/classification_datasets/v8'
DST_DIR.mkdir(exist_ok=True, parents=True)

# SAGITTAL_CROP_MM = [45, 50]  # h:45mm, w:50mm
# AXIAL_CROP_MM = [45, 75]  # h:40mm, w:75mm
SAGITTAL_CROP_MM = [40, 60]  # h:40mm, w:60mm
AXIAL_CROP_MM = [80, 70]  # h:80mm, w:65mm

# キーポイント検出器
SAGITTAL_KEYPOINT_TRAIN_CSV = SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v6/train_5folds.csv'
SAGITTAL_KEYPOINT_MODEL_INFO_F0 = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0002_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0002/exp0002_fold0_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0003/exp0003_fold0_best_score.pth',
    },
]
SAGITTAL_KEYPOINT_MODEL_INFO_F1 = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0002_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0002/exp0002_fold1_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0003/exp0003_fold1_best_score.pth',
    },
]
SAGITTAL_KEYPOINT_MODEL_INFO_F2 = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0002_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0002/exp0002_fold2_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0003/exp0003_fold2_best_score.pth',
    },
]
SAGITTAL_KEYPOINT_MODEL_INFO_F3 = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0002_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0002/exp0002_fold3_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0003/exp0003_fold3_best_score.pth',
    },
]
SAGITTAL_KEYPOINT_MODEL_INFO_F4 = [
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0002_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0002/exp0002_fold4_best_score.pth',
    },
    {
        'config': f'src/stage1/{args.user}/sagittal/config/v6_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v6/0003/exp0003_fold4_best_score.pth',
    },
]

AXIAL_KEYPOINT_TRAIN_CSV = SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v5/train_10folds.csv'
AXIAL_KEYPOINT_MODEL_INFO_F0 = [
    {
        'config': f'src/stage1/{args.user}/axial/config/v5_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v5/0003/exp0003_fold0_best_score.pth',
    },
]
AXIAL_KEYPOINT_MODEL_INFO_F1 = [
    {
        'config': f'src/stage1/{args.user}/axial/config/v5_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v5/0003/exp0003_fold1_best_score.pth',
    },
]
AXIAL_KEYPOINT_MODEL_INFO_F2 = [
    {
        'config': f'src/stage1/{args.user}/axial/config/v5_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v5/0003/exp0003_fold2_best_score.pth',
    },
]
AXIAL_KEYPOINT_MODEL_INFO_F3 = [
    {
        'config': f'src/stage1/{args.user}/axial/config/v5_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v5/0003/exp0003_fold3_best_score.pth',
    },
]
AXIAL_KEYPOINT_MODEL_INFO_F4 = [
    {
        'config': f'src/stage1/{args.user}/axial/config/v5_0003_config.yaml',
        'checkpoint': SETTINGS.model_checkpoint_dir / f'stage1/{args.user}/v5/0003/exp0003_fold4_best_score.pth',
    },
]

CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
SEGMENTS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
PLANES = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']

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


def convert_world_to_image_coord(
    image_position: list,
    image_orientation: list,
    pixel_spacing: list,
    px: float,
    py: float,
    pz: float,
) -> tuple[int, int]:
    sx = image_position[0]
    sy = image_position[1]
    sz = image_position[2]

    # 3Dの変換行列
    xx_di = image_orientation[0] * pixel_spacing[0]
    xy_di = image_orientation[1] * pixel_spacing[0]
    xz_di = image_orientation[2] * pixel_spacing[0]
    yx_dj = image_orientation[3] * pixel_spacing[1]
    yy_dj = image_orientation[4] * pixel_spacing[1]
    yz_dj = image_orientation[5] * pixel_spacing[1]

    # 3次元変換行列
    transform_matrix = np.array([
        [xx_di, yx_dj],
        [xy_di, yy_dj],
        [xz_di, yz_dj]
    ])

    # 実座標から原点を引く
    world_vector = np.array([px - sx, py - sy, pz - sz])

    # 転置行列を使って画像座標を得る（2次元に戻す）
    inverse_matrix = np.linalg.pinv(transform_matrix)  # 疑似逆行列を使用
    image_coord = np.dot(inverse_matrix, world_vector)

    return round(image_coord[0]), round(image_coord[1])

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
# Function to calculate the midpoint between two points


def midpoint(p1, p2):
    return tuple((np.array(p1) + np.array(p2)) / 2)


# Function to calculate an extended point
def extended_point(p1, p2, direction='outward'):
    vector = np.array(p2) - np.array(p1)
    extended = np.array(p1) - vector
    return tuple(extended)


def extend_level_point(points):
    # Midpoints between each level
    midpoints = {
        'L1/L2-L2/L3': midpoint(points['L1/L2'], points['L2/L3']),
        'L2/L3-L3/L4': midpoint(points['L2/L3'], points['L3/L4']),
        'L3/L4-L4/L5': midpoint(points['L3/L4'], points['L4/L5']),
        'L4/L5-L5/S1': midpoint(points['L4/L5'], points['L5/S1']),
    }

    # Calculate the outward extended points
    extended_upper = extended_point(points['L1/L2'], midpoints['L1/L2-L2/L3'])
    extended_lower = extended_point(points['L5/S1'], midpoints['L4/L5-L5/S1'])

    # Combine all points
    extend_points = {
        'L1/L2_upper': extended_upper,
        'L1/L2-L2/L3_mid': midpoints['L1/L2-L2/L3'],
        'L2/L3-L3/L4_mid': midpoints['L2/L3-L3/L4'],
        'L3/L4-L4/L5_mid': midpoints['L3/L4-L4/L5'],
        'L4/L5-L5/S1_mid': midpoints['L4/L5-L5/S1'],
        'L5/S1_lower': extended_lower
    }
    return extend_points


def point_on_plane(plane_points, normal, point):
    p1, p2, p3 = plane_points[:3]
    r = np.array(point) - np.array(p1)
    return np.dot(normal, r) == 0


def line_plane_intersection(plane_points, normal, line_points):
    p1, p2 = line_points
    line_vector = np.array(p2) - np.array(p1)
    dot_product = np.dot(normal, line_vector)
    if dot_product == 0:
        # The line is parallel to the plane
        return point_on_plane(plane_points, normal, p1)

    # Find the parameter t for the intersection% point
    t = np.dot(normal, np.array(plane_points[0]) - np.array(p1)) / dot_product

    # Check if the intersection point lies on the line level
    return 0 <= t <= 1

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

# %% [markdown]
# # Reading and Taking a look csv


# %%
DESC_DF = pd.read_csv(f'{ROOT_DIR}/train_series_descriptions.csv')
DESC_DF.head()

# %%
COORD_DF = pd.read_csv(SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v6/train_label_coordinates_fixed.csv')
COORD_DF.head()

# %%
SAGITTAL_KPT_TRAIN_DF = pd.read_csv(SAGITTAL_KEYPOINT_TRAIN_CSV)
SAGITTAL_KPT_TRAIN_DF = SAGITTAL_KPT_TRAIN_DF[SAGITTAL_KPT_TRAIN_DF['source'] == 'rsna']
SAGITTAL_KPT_TRAIN_DF.head()

# %%
AXIAL_KPT_TRAIN_DF = pd.read_csv(AXIAL_KEYPOINT_TRAIN_CSV)
AXIAL_KPT_TRAIN_DF = AXIAL_KPT_TRAIN_DF[AXIAL_KPT_TRAIN_DF['source'] == 'rsna']
AXIAL_KPT_TRAIN_DF.head()

# %% [markdown]
# ## キーポイント検出器

# %%


def get_config(config_path: str, dot_list: list) -> dict:
    config_omega_from_yaml = OmegaConf.load(config_path)
    config_omega_from_args = OmegaConf.from_dotlist(dot_list)
    config_omega = OmegaConf.merge(config_omega_from_yaml, config_omega_from_args)
    config = OmegaConf.to_container(config_omega, resolve=True)  # DictConfig -> dict
    return config

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


class RSNA2024SagittalKeypointDetector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 model_class=RSNA2024KeypointNetTest,
                 levels: list = SEGMENTS,
                 condition: list = CONDITIONS,
                 planes: list = PLANES):
        self._model_class = model_class
        self._model_infos = model_infos
        self._device = device
        self._levels = levels
        self._condition = condition
        self._planes = planes
        self._models = []
        self._transforms = []
        self._input_channels = []

        for model_info in model_infos:
            config_path = model_info['config']
            checkpoint = model_info['checkpoint']

            config = get_config(config_path, [])
            config['model']['timm_model']['pretrained'] = False
            model = model_class(**config['model'])
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            model.to(device)
            self._models.append(model)
            self._transforms.append(A.Compose([
                A.Resize(*config['transform']['image_size']),
                A.Normalize(mean=0.5, std=0.5)
            ]))
            self._input_channels.append(config['dataset'].get('num_slices', 1))

    def _select_n_elements(self, lst: np.ndarray, n: int, base_index: int) -> np.ndarray:
        length = len(lst)

        # 基準位置からn個のインデックスを生成
        offset = n // 2
        indices = np.linspace(base_index - offset, base_index + offset, n).astype(int)

        # インデックスを 0 以上、length-1 以下にクリップ
        indices = np.clip(indices, 0, length - 1)

        # 対応する要素を返す
        return lst[indices]

    def _preprocess(self, stack: dict, transform: A.Compose, input_channel: int, device: torch.device) -> torch.Tensor:
        num_slices = len(stack['instance_numbers'])
        indices = self._select_n_elements(np.arange(num_slices), input_channel, num_slices // 2)
        slices = []
        for idx in indices:
            image = apply_min_max_norm(stack['arrays'][idx])
            image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            slices.append(image)
        image = np.stack(slices, axis=-1)

        transformed_image = transform(image=image)['image']
        transformed_image = transformed_image.transpose(2, 0, 1)
        transformed_image = torch.Tensor(transformed_image[None, ...]).to(device)
        return transformed_image

    def _postprocess(self,
                     heatmap: np.ndarray,
                     image_shape: list,
                     image_position_patient: list,
                     image_orientation_patient: list,
                     pixel_spacings: list,
                     keypoint_threshold: float) -> tuple:
        image_coord = {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None}
        image_norm_coord = {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None}
        world_coord = {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None}

        hmap_height, hmap_width = heatmap.shape[-2:]
        for i, level in enumerate(image_norm_coord.keys()):
            if heatmap[0][i].max() < keypoint_threshold:
                continue
            threshold = np.percentile(heatmap[0][i], 99.8)
            mask = heatmap[0][i] >= threshold
            mask_indexes = np.where(mask)
            y_min = np.min(mask_indexes[0]) + 0.5
            y_max = np.max(mask_indexes[0]) + 0.5
            x_min = np.min(mask_indexes[1]) + 0.5
            x_max = np.max(mask_indexes[1]) + 0.5
            hmap_x, hmap_y = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            norm_x, norm_y = (hmap_x / hmap_width, hmap_y / hmap_height)
            x, y = (norm_x * image_shape[1], norm_y * image_shape[0])
            image_coord[level] = (x, y)
            image_norm_coord[level] = (norm_x, norm_y)
            world_coord[level] = convert_image_to_world_coord(image_position_patient,
                                                              image_orientation_patient,
                                                              pixel_spacings,
                                                              x, y)

        return image_coord, image_norm_coord, world_coord

    def predict(
        self,
        stack: Optional[dict],
        threshold: float = 0.0,
    ) -> dict[str, list[float]]:
        if stack is None:
            return ({"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None},
                    {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None},
                    {"L1/L2": None, "L2/L3": None, "L3/L4": None, "L4/L5": None, "L5/S1": None})
        num_slices = len(stack['instance_numbers'])
        stack_index = num_slices // 2

        heatmap_list = []
        for model, transform, input_channel in zip(self._models, self._transforms, self._input_channels):
            processed_image = self._preprocess(stack, transform, input_channel, self._device)
            with torch.no_grad():
                logit = model(processed_image)['logits']
            heatmap = logit.sigmoid().detach().cpu().numpy()
            heatmap_list.append(heatmap)
        heatmap = np.concatenate(heatmap_list, axis=0).mean(axis=0, keepdims=True)

        image_shape = stack['image_shapes'][stack_index]
        image_position_patient = stack['image_position_patients'][stack_index]
        image_orientation_patient = stack['image_orientation_patients'][stack_index]
        pixel_spacings = stack['pixel_spacings'][stack_index]
        image_coord, image_norm_coord, world_coord = self._postprocess(heatmap, image_shape,
                                                                       image_position_patient, image_orientation_patient,
                                                                       pixel_spacings, threshold)

        return image_coord, image_norm_coord, world_coord

# %%


class RSNA2024AxialKeypointDetector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 model_class=RSNA2024KeypointNetTest,
                 levels: list = SEGMENTS,
                 condition: list = CONDITIONS,
                 planes: list = PLANES):
        self._model_class = model_class
        self._model_infos = model_infos
        self._device = device
        self._levels = levels
        self._condition = condition
        self._planes = planes
        self._models = []
        self._transforms = []
        self._input_channels = []

        for model_info in model_infos:
            config_path = model_info['config']
            checkpoint = model_info['checkpoint']

            config = get_config(config_path, [])
            config['model']['timm_model']['pretrained'] = False
            model = model_class(**config['model'])
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            model.to(device)
            self._models.append(model)
            self._transforms.append(A.Compose([
                A.Resize(*config['transform']['image_size']),
                A.Normalize(mean=0.5, std=0.5)
            ]))
            self._input_channels.append(config['dataset'].get('num_slices', 1))

    def _select_n_elements(self, lst: np.ndarray, n: int, base_index: int) -> np.ndarray:
        length = len(lst)

        # 基準位置からn個のインデックスを生成
        offset = n // 2
        indices = np.linspace(base_index - offset, base_index + offset, n).astype(int)

        # インデックスを 0 以上、length-1 以下にクリップ
        indices = np.clip(indices, 0, length - 1)

        # 対応する要素を返す
        return lst[indices]

    def _preprocess(self, stack: dict, transform: A.Compose, input_channel: int, device: torch.device) -> torch.Tensor:
        num_slices = len(stack['instance_numbers'])
        indices = self._select_n_elements(np.arange(num_slices), input_channel, num_slices // 2)
        slices = []
        for idx in indices:
            image = apply_min_max_norm(stack['arrays'][idx])
            image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            slices.append(image)
        image = np.stack(slices, axis=-1)

        transformed_image = transform(image=image)['image']
        transformed_image = transformed_image.transpose(2, 0, 1)
        transformed_image = torch.Tensor(transformed_image[None, ...]).to(device)
        return transformed_image

    def _postprocess(self,
                     heatmap: np.ndarray,
                     image_shape: list,
                     image_position_patient: list,
                     image_orientation_patient: list,
                     pixel_spacings: list,
                     keypoint_threshold: float) -> tuple:
        image_coord = {"center": None}
        image_norm_coord = {"center": None}
        world_coord = {"center": None}

        hmap_height, hmap_width = heatmap.shape[-2:]
        for i, level in enumerate(image_norm_coord.keys()):
            if heatmap[0][i].max() < keypoint_threshold:
                continue
            threshold = np.percentile(heatmap[0][i], 99.8)
            mask = heatmap[0][i] >= threshold
            mask_indexes = np.where(mask)
            y_min = np.min(mask_indexes[0]) + 0.5
            y_max = np.max(mask_indexes[0]) + 0.5
            x_min = np.min(mask_indexes[1]) + 0.5
            x_max = np.max(mask_indexes[1]) + 0.5
            hmap_x, hmap_y = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            norm_x, norm_y = (hmap_x / hmap_width, hmap_y / hmap_height)
            x, y = (norm_x * image_shape[1], norm_y * image_shape[0])
            image_coord[level] = (x, y)
            image_norm_coord[level] = (norm_x, norm_y)
            world_coord[level] = convert_image_to_world_coord(image_position_patient,
                                                              image_orientation_patient,
                                                              pixel_spacings,
                                                              x, y)

        return image_coord, image_norm_coord, world_coord

    def predict(
        self,
        stack: Optional[dict],
        threshold: float = 0.0,
    ) -> dict[str, list[float]]:
        if stack is None:
            return ({"center": None},
                    {"center": None},
                    {"center": None})
        num_slices = len(stack['instance_numbers'])
        stack_index = num_slices // 2

        heatmap_list = []
        for model, transform, input_channel in zip(self._models, self._transforms, self._input_channels):
            processed_image = self._preprocess(stack, transform, input_channel, self._device)
            with torch.no_grad():
                logit = model(processed_image)['logits']
            heatmap = logit.sigmoid().detach().cpu().numpy()
            heatmap_list.append(heatmap)
        heatmap = np.concatenate(heatmap_list, axis=0).mean(axis=0, keepdims=True)

        image_shape = stack['image_shapes'][stack_index]
        image_position_patient = stack['image_position_patients'][stack_index]
        image_orientation_patient = stack['image_orientation_patients'][stack_index]
        pixel_spacings = stack['pixel_spacings'][stack_index]
        image_coord, image_norm_coord, world_coord = self._postprocess(heatmap, image_shape,
                                                                       image_position_patient, image_orientation_patient,
                                                                       pixel_spacings, threshold)

        return image_coord, image_norm_coord, world_coord


# %%
SAGITTAL_KEYPOINT_DETECTORS = [
    RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO_F0, device=DEVICE),
    RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO_F1, device=DEVICE),
    RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO_F2, device=DEVICE),
    RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO_F3, device=DEVICE),
    RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO_F4, device=DEVICE),
]
AXIAL_KEYPOINT_DETECTORS = [
    RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO_F0, device=DEVICE),
    RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO_F1, device=DEVICE),
    RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO_F2, device=DEVICE),
    RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO_F3, device=DEVICE),
    RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO_F4, device=DEVICE),
]

# %% [markdown]
# # PNG, CSV作成
#
# 各studyが参照する画像を記録したCSVファイルを作成する。

# %%


def get_sagittal_coord_from_gt(keypoint_train_info: dict, stack: dict) -> tuple:
    image_coord = {}
    image_norm_coord = {}
    world_coord = {}

    # 予測結果が不正確な場合はGTを使用
    instance_number = keypoint_train_info['instance_number']
    gt_idx = np.where(stack['instance_numbers'] == instance_number)[0][0]
    image_position = stack['image_position_patients'][gt_idx]
    image_orientation = stack['image_orientation_patients'][gt_idx]
    pixel_spacing = stack['pixel_spacings'][gt_idx]
    image_shape = stack['image_shapes'][gt_idx]

    for level in SEGMENTS:
        image_coord[level] = (keypoint_train_info[f'{level}_x'], keypoint_train_info[f'{level}_y'])
        image_norm_coord[level] = (keypoint_train_info[f'{level}_x'] / image_shape[1], keypoint_train_info[f'{level}_y'] / image_shape[0])
        world_coord[level] = convert_image_to_world_coord(image_position,
                                                          image_orientation,
                                                          pixel_spacing,
                                                          keypoint_train_info[f'{level}_x'], keypoint_train_info[f'{level}_y'])
    return image_coord, image_norm_coord, world_coord


def validate_sagittal_keypoint(preds: np.ndarray, targets: np.ndarray, normalization_constant: float, threshold: float = 0.04) -> bool:
    distances = np.linalg.norm(targets - preds, axis=1)
    norm_distance = distances / normalization_constant
    return np.all(norm_distance < threshold, axis=0)

# %% [markdown]
# study_id	series_id	instance_number	condition	level	x	y
# 0	4003253	702807833	8	Spinal Canal Stenosis	L1/L2	322.831858	227.964602
# 1	4003253	702807833	8	Spinal Canal Stenosis	L2/L3	320.571429	295.714286
# 2	4003253	702807833	8	Spinal Canal Stenosis	L3/L4	323.030303	371.818182
# 3	4003253	702807833	8	Spinal Canal Stenosis	L4/L5	335.292035	427.327434
# 4	4003253	702807833	8	Spinal Canal Stenosis	L5/S1	353.415929	483.964602


# %%
data = []
st_ids = DESC_DF['study_id'].unique()
descs = list(DESC_DF['series_description'].unique())
sagittal_kpt_ng_studies = []
crop_data = []

all_level_studies = []
for idx, study_id in enumerate(tqdm(st_ids, total=len(st_ids))):
    pdf = DESC_DF[DESC_DF['study_id'] == study_id]
    study_sagittal_kpt_train_df = SAGITTAL_KPT_TRAIN_DF[SAGITTAL_KPT_TRAIN_DF['study_id'] == study_id]

    desc_to_series_id = {}
    for desc in descs:
        series_ids = pdf[pdf['series_description'] == desc]['series_id'].values.tolist()
        desc_to_series_id[desc] = series_ids

    stacks = {series_id: None for series_ids in desc_to_series_id.values() for series_id in series_ids}
    for desc, series_ids in desc_to_series_id.items():
        if len(series_ids) == 0:
            print(study_id, desc, 'has no images')
            continue

        for series_id in series_ids:
            dicom_folder = f'{ROOT_DIR}/train_images/{study_id}/{series_id}'
            stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])
            stacks[series_id] = stack

    # Sagittal
    sagittal_world_coords = {'Sagittal T1': [], 'Sagittal T2/STIR': []}
    for desc in ['Sagittal T1', 'Sagittal T2/STIR']:
        for series_id in desc_to_series_id[desc]:
            stack = stacks[series_id]
            if stack is None:
                continue

            # あんまうまくいかなかった
            # if sagittal_predict_series_id != series_id:
            #     center_slice = len(stack['instance_numbers']) // 2
            #     image_shape = stack['image_shapes'][center_slice]
            #     image_position = stack['image_position_patients'][center_slice]
            #     image_orientation = stack['image_orientation_patients'][center_slice]
            #     pixel_spacing = stack['pixel_spacings'][center_slice]
            #     crop_image_norm_coord = {}
            #     for level, position in sagittal_world_coord.items():
            #         image_coord = convert_world_to_image_coord(image_position, image_orientation, pixel_spacing, *position)
            #         crop_image_norm_coord[level] = (image_coord[0] / image_shape[1], image_coord[1] / image_shape[0])
            # else:
            #     crop_image_norm_coord = sagittal_image_norm_coord

            if len(study_sagittal_kpt_train_df) == 0:
                sagittal_kpt_detector = SAGITTAL_KEYPOINT_DETECTORS[0]
            else:
                sagittal_kpt_detector = SAGITTAL_KEYPOINT_DETECTORS[study_sagittal_kpt_train_df.iloc[0]['fold']]
            sagittal_image_coord, sagittal_image_norm_coord, sagittal_world_coord = sagittal_kpt_detector.predict(stacks[series_id])

            # もしGTデータがあるなら、予測とGTの距離が一定範囲内に収まっているか検証して、収まっていない場合はGTを使用する
            series_sagittal_kpt_train_df = study_sagittal_kpt_train_df[study_sagittal_kpt_train_df['series_id'] == series_id]
            if len(series_sagittal_kpt_train_df) == 1:
                keypoint_train_info = series_sagittal_kpt_train_df.iloc[0].to_dict()
                is_ok = validate_sagittal_keypoint(np.array([sagittal_image_coord[f'{level}'] for level in SEGMENTS]),
                                                   np.array([(keypoint_train_info[f'{level}_x'], keypoint_train_info[f'{level}_y']) for level in SEGMENTS]),
                                                   max(keypoint_train_info['height'], keypoint_train_info['width']))
                if not is_ok:
                    # 予測結果がGTと乖離している場合はGTを使用
                    sagittal_image_coord, sagittal_image_norm_coord, sagittal_world_coord = get_sagittal_coord_from_gt(keypoint_train_info, stacks[series_id])
                    sagittal_kpt_ng_studies.append(study_id)
                    print(study_id, 'prediction is not accurate')

            sagittal_world_coords[desc].append(sagittal_world_coord)

            num_slices = len(stack['instance_numbers'])
            for i in list(range(num_slices)):
                image = apply_min_max_norm(stack['arrays'][i])
                pixel_spacing = stack['pixel_spacings'][i]
                crop_y_half, crop_x_half = (SAGITTAL_CROP_MM / pixel_spacing / 2).tolist()
                image_shape = stack['image_shapes'][i]
                for level, position in sagittal_image_norm_coord.items():
                    x, y = position[0] * image_shape[1], position[1] * image_shape[0]
                    left = round(x - crop_x_half)
                    right = round(x + crop_x_half)
                    top = round(y - crop_y_half)
                    bottom = round(y + crop_y_half)
                    if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
                        print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
                        continue
                    crop_image = image[top:bottom, left:right]
                    level_path = level.replace('/', '-')
                    png_path = DST_DIR / f'train_images_png/{study_id}/{series_id}/{level_path}/{stack["instance_numbers"][i]}.png'
                    if not png_path.exists():
                        png_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(png_path, crop_image)
                    data.append([study_id, series_id, desc, level, stack['instance_numbers'][i], stack['standard_positions'][i], png_path.relative_to(DST_DIR)])

                    # 分析用にクロップ情報を保存
                    crop_data.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'instance_number': stack['instance_numbers'][i],
                        'level': level,
                        'x': x,
                        'y': y,
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'series_description': desc,
                    })

    # Axial
    # サジタルT2のキーポイントから算出した実座標を用いたしてアキシャルのグルーピングを行う
    # (サジタルT2がない場合はT1を使用する)
    if len(sagittal_world_coords['Sagittal T2/STIR']):
        sagittal_world_coord = sagittal_world_coords['Sagittal T2/STIR'][0]
    else:
        sagittal_world_coord = sagittal_world_coords['Sagittal T1'][0]

    points = extend_level_point(sagittal_world_coord)
    lines = {'L1/L2': [points['L1/L2_upper'], points['L1/L2-L2/L3_mid']],
             'L2/L3': [points['L1/L2-L2/L3_mid'], points['L2/L3-L3/L4_mid']],
             'L3/L4': [points['L2/L3-L3/L4_mid'], points['L3/L4-L4/L5_mid']],
             'L4/L5': [points['L3/L4-L4/L5_mid'], points['L4/L5-L5/S1_mid']],
             'L5/S1': [points['L4/L5-L5/S1_mid'], points['L5/S1_lower']]}

    study_axial_kpt_train_df = AXIAL_KPT_TRAIN_DF[AXIAL_KPT_TRAIN_DF['study_id'] == study_id]
    if len(study_axial_kpt_train_df) == 0:
        axial_kpt_detector = AXIAL_KEYPOINT_DETECTORS[0]
    else:
        fold = study_axial_kpt_train_df.iloc[0]['fold']
        if fold < 5:
            axial_kpt_detector = AXIAL_KEYPOINT_DETECTORS[fold]
        else:
            axial_kpt_detector = AXIAL_KEYPOINT_DETECTORS[0]

    for desc in ['Axial T2']:
        for series_id in desc_to_series_id[desc]:
            stack = stacks[series_id]
            if stack is None:
                continue

            num_slices = len(stack['instance_numbers'])
            ipp = stack['image_position_patients']
            iop = stack['image_orientation_patients']
            ps = stack['pixel_spacings']
            image_shapes = stack['image_shapes']
            p0s = convert_image_to_world_coord_array(ipp, iop, ps, np.zeros(num_slices), np.zeros(num_slices))
            p1s = convert_image_to_world_coord_array(ipp, iop, ps, image_shapes[..., 0], np.zeros(num_slices))
            p2s = convert_image_to_world_coord_array(ipp, iop, ps, np.zeros(num_slices), image_shapes[..., 1])
            p3s = convert_image_to_world_coord_array(ipp, iop, ps, image_shapes[..., 0], image_shapes[..., 1])
            planes = np.array([p0s, p1s, p2s])
            normals = np.cross(p1s - p0s, p2s - p0s)

            stack_indices_per_level = {level: [] for level in lines.keys()}
            for i in range(num_slices):
                plane = [p0s[i], p1s[i], p2s[i]]
                normal = normals[i]
                for level, line in lines.items():
                    if line_plane_intersection(plane, normal, line):
                        stack_indices_per_level[level].append(i)
                        break

            for level, stack_indices in stack_indices_per_level.items():
                if len(stack_indices) == 0:
                    continue

                axial_image_coord, axial_image_norm_coord, axial_world_coord = axial_kpt_detector.predict({
                    'instance_numbers': stack['instance_numbers'][stack_indices],
                    'arrays': [stack['arrays'][i] for i in stack_indices],
                    'image_shapes': stack['image_shapes'][stack_indices],
                    'image_position_patients': stack['image_position_patients'][stack_indices],
                    'image_orientation_patients': stack['image_orientation_patients'][stack_indices],
                    'pixel_spacings': stack['pixel_spacings'][stack_indices],
                    'standard_positions': stack['standard_positions'][stack_indices],
                })
                level_world_position = sagittal_world_coord[level]
                for i in stack_indices:
                    image = apply_min_max_norm(stack['arrays'][i])
                    image_shape = stack['image_shapes'][i]
                    image_position = stack['image_position_patients'][i]
                    image_orientation = stack['image_orientation_patients'][i]
                    pixel_spacing = stack['pixel_spacings'][i]
                    crop_y_half, crop_x_half = (AXIAL_CROP_MM / pixel_spacing / 2).tolist()

                    x, y = axial_image_norm_coord['center'][0] * image_shape[1], axial_image_norm_coord['center'][1] * image_shape[0]
                    left = round(x - crop_x_half)
                    right = round(x + crop_x_half)
                    top = round(y - crop_y_half)
                    bottom = round(y + crop_y_half)
                    if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
                        print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
                        continue

                    crop_image = image[top:bottom, left:right]
                    level_path = level.replace('/', '-')
                    png_path = DST_DIR / f'train_images_png/{study_id}/{series_id}/{level_path}/{stack["instance_numbers"][i]}.png'
                    if not png_path.exists():
                        png_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(png_path, crop_image)
                    data.append([study_id, series_id, desc, level, stack['instance_numbers'][i], stack['standard_positions'][i], png_path.relative_to(DST_DIR)])

                    # 分析用にクロップ情報を保存
                    crop_data.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'instance_number': stack['instance_numbers'][i],
                        'level': level,
                        'x': x,
                        'y': y,
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'series_description': desc,
                    })

train_image_df = pd.DataFrame(data=data, columns=['study_id', 'series_id', 'series_description', 'level', 'instance_no', 'standard_pos', 'image_path'])
train_image_df = train_image_df.drop_duplicates()
train_image_df.to_csv(DST_DIR / 'train_images.csv', index=False)
print(f'ng={len(sagittal_kpt_ng_studies) / 1975 * 100:.2f}%')

# 分析用
CROP_DF = pd.DataFrame(crop_data)
CROP_DF.sort_values(['study_id', 'series_id', 'instance_number', 'level'], inplace=True)
CROP_DF.reset_index(drop=True, inplace=True)

# %% [markdown]

# %%
# Sagittalは各シリーズにレベルが割り振られるためシリーズ単位で解析する
_crop_df = CROP_DF[CROP_DF['series_description'].isin(['Sagittal T1', 'Sagittal T2/STIR'])].copy()
_crop_df['height'] = _crop_df['bottom'] - _crop_df['top']
_crop_df['width'] = _crop_df['right'] - _crop_df['left']

study_ids = _crop_df['study_id'].unique()
total = tp = fp = 0
for study_id in tqdm(study_ids, total=len(study_ids)):
    study_crop_df = _crop_df[_crop_df['study_id'] == study_id]
    study_gt_coord_df = COORD_DF[COORD_DF['study_id'] == study_id]
    for series_id in study_crop_df['series_id'].unique():
        series_gt_coord_df = study_gt_coord_df[study_gt_coord_df['series_id'] == series_id]

        series_crop_df = study_crop_df[study_crop_df['series_id'] == series_id]
        instance_number = series_crop_df['instance_number'].unique()[series_crop_df['instance_number'].nunique() // 2]
        series_crop_df = series_crop_df[series_crop_df['instance_number'] == instance_number]
        desc = series_crop_df['series_description'].values[0]

        for i, gt_row in series_gt_coord_df.iterrows():
            series_crop_df['norm_x'] = (gt_row['x'] - series_crop_df['left']) / series_crop_df['width']
            series_crop_df['norm_y'] = (gt_row['y'] - series_crop_df['top']) / series_crop_df['height']
            series_crop_df['in_crop'] = (series_crop_df['norm_x'] >= 0) & (series_crop_df['norm_x'] <=
                                                                           1) & (series_crop_df['norm_y'] >= 0) & (series_crop_df['norm_y'] <= 1)
            # クロップした位置に正しい椎間板レベルのGTが含まれているならTP
            tp += int(np.all(series_crop_df[series_crop_df['level'] == gt_row['level']]['in_crop']))
            # クロップした位置に正しくない椎間板レベルのGTが含まれているならFP
            fp += int(np.any(series_crop_df[series_crop_df['level'] != gt_row['level']]['in_crop']))
            total += 1

print(f'GTが正しい椎間板レベルのスライスに含まれる確率={tp / total * 100:.2f}% ({tp}/{total})')
print(f'GTが誤った椎間板レベルのスライスに含まれる確率={fp / total * 100:.2f}% ({fp}/{total})')

# %%
# Axialは各インスタンスにレベルが割り振られるためインスタンス単位で解析する
crop_analysis_data = []

_crop_df = CROP_DF[CROP_DF['series_description'].isin(['Axial T2'])].copy()
study_ids = _crop_df['study_id'].unique()

for study_id in tqdm(study_ids, total=len(study_ids)):
    study_crop_df = _crop_df[_crop_df['study_id'] == study_id]
    study_gt_coord_df = COORD_DF[COORD_DF['study_id'] == study_id]
    for series_id in study_crop_df['series_id'].unique():
        series_crop_df = study_crop_df[study_crop_df['series_id'] == series_id]
        series_gt_coord_df = study_gt_coord_df[study_gt_coord_df['series_id'] == series_id]
        desc = series_crop_df['series_description'].values[0]

        for _, gt_row in series_gt_coord_df.iterrows():
            instance_number = gt_row.instance_number
            instance_crop_df = series_crop_df[series_crop_df['instance_number'] == instance_number]
            if len(instance_crop_df) == 0:
                crop_analysis_data.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'instance_number': instance_number,
                    'level': None,
                    'series_description': desc,
                    'condition': gt_row.condition,
                    'gt_level': gt_row.level,
                    'gt_norm_x': None,
                    'gt_norm_y': None,
                    'same_level': None,
                    'exist': False,
                })
            elif len(instance_crop_df) == 1:
                row = instance_crop_df.iloc[0]
                level = row.level
                left, top, right, bottom = row.left, row.top, row.right, row.bottom
                height, width = bottom - top, right - left

                # クロップ位置に対するgtの正規化座標を算出
                gt_norm_x = (gt_row.x - left) / width
                gt_norm_y = (gt_row.y - top) / height
                condition = gt_row.condition
                crop_analysis_data.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'instance_number': instance_number,
                    'level': level,
                    'series_description': desc,
                    'condition': condition,
                    'gt_level': gt_row.level,
                    'gt_norm_x': gt_norm_x,
                    'gt_norm_y': gt_norm_y,
                    'same_level': level == gt_row.level,
                    'exist': True,
                })
            else:
                raise ValueError('Invalid instance_number')

crop_analysis_df = pd.DataFrame(data=crop_analysis_data)
crop_analysis_df = crop_analysis_df.drop_duplicates()

print(f"スライスの取りこぼし確率={(1 - crop_analysis_df['exist'].sum() / len(crop_analysis_df)) * 100:.2f}% ({crop_analysis_df['exist'].sum()}/{len(crop_analysis_df)})")
exist_crop_analysis_df = crop_analysis_df[crop_analysis_df['exist']]
print(
    f"GTが正しい椎間板レベルのスライスに含まれる確率={(exist_crop_analysis_df['same_level'] == True).sum() / len(exist_crop_analysis_df) * 100:.2f}% ({(exist_crop_analysis_df['same_level'] == True).sum()}/{len(exist_crop_analysis_df)})")
print(
    f"GTが誤った椎間板レベルのスライスに含まれる確率={(exist_crop_analysis_df['same_level'] == False).sum() / len(exist_crop_analysis_df) * 100:.2f}% ({(exist_crop_analysis_df['same_level'] == False).sum()}/{len(exist_crop_analysis_df)})")
