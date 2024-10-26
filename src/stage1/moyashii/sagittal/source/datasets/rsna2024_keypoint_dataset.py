# reference: https://www.kaggle.com/code/hugowjd/rsna2024-lsdc-training-densenet#Define-Dataset
# reference: https://github.com/xingyizhou/CenterNet

import math
from typing import Optional

import os
import re
import glob

import cv2
import pydicom
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations as A

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.datasets.dataset_phase import DatasetPhase  # noqa


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = round(center[0]), round(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class RSNA2024KeypointDatasetTrainV1(Dataset):
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        transform: A.Compose = None,
        heatmap_size: tuple[int, int] = (20, 20),
        stride: int = 4,
    ) -> None:
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        self._phase = phase
        self._transform = transform
        self._heatmap_size = heatmap_size
        self._labels = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4,
        }
        self._stride = stride

    def __len__(self):
        return len(self._train_df)

    def _read_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.astype(np.uint8)

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        study_id = target_row['study_id']
        image_path = target_row['png_path']
        image = self._read_image((self._image_root / image_path).as_posix())

        keypoints = []
        class_labels = []
        for segment in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
            seg_x = segment + '_x'
            seg_y = segment + '_y'
            keypoints.append((round(target_row[seg_x]), round(target_row[seg_y])))
            class_labels.append(segment)

        if self._transform is not None:
            transformed = self._transform(image=image, keypoints=keypoints, class_labels=class_labels)
            image = transformed['image']
            keypoints = transformed['keypoints']
            class_labels = transformed['class_labels']

        h, w = image.shape[:2]
        heatmap = np.zeros([len(self._labels), h // self._stride, w // self._stride])
        radius = gaussian_radius([s // self._stride for s in self._heatmap_size])
        radius = max(0, int(radius))
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            kpt_x = keypoint[0] / self._stride
            kpt_y = keypoint[1] / self._stride
            draw_gaussian(heatmap[index], (kpt_x, kpt_y), radius)

        image = image.astype(np.float32)
        image = image[None, ...]

        ret_keypoints = [(-1.0, -1.0) for _ in range(len(self._labels))]
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            ret_keypoints[index] = keypoint
        return image, heatmap, study_id, np.asarray(ret_keypoints)


class RSNA2024KeypointDatasetTrainV2(Dataset):
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        transform: A.Compose = None,
        heatmap_size: tuple[int, int] = (20, 20),
        num_slices: int = 1,
        use_center: bool = False,
        stride: int = 4,
    ) -> None:
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        self._phase = phase
        self._transform = transform
        self._heatmap_size = heatmap_size
        self._num_slices = num_slices
        if self._num_slices % 2 == 0:
            raise ValueError('num_slices must be an odd number.')
        self._use_center = use_center
        self._stride = stride
        self._labels = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4,
        }

    def __len__(self):
        return len(self._train_df)

    def _read_image(self, image_path: str, image_size: Optional[tuple[int, int]] = (512, 512)) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_size is not None:
            image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_LINEAR)
        return image.astype(np.uint8)

    def _select_n_elements(self, lst: np.ndarray, n: int, base_index: int) -> np.ndarray:
        length = len(lst)

        # 基準位置からn個のインデックスを生成
        offset = n // 2
        indices = np.linspace(base_index - offset, base_index + offset, n).astype(int)

        # インデックスを 0 以上、length-1 以下にクリップ
        indices = np.clip(indices, 0, length - 1)

        # 対応する要素を返す
        return lst[indices]

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        study_id = target_row['study_id']

        image_dir = self._image_root / target_row['image_dir']
        image_paths = np.asarray(sorted(image_dir.iterdir()))
        base_index = -1
        base_instance_number = target_row['instance_number']
        for i, image_path in enumerate(image_paths):
            instance_number = int(image_path.stem.split('_')[1])
            if instance_number == base_instance_number:
                base_index = i
                break
        assert base_index != -1, f'base_index is not found. study_id: {study_id}, base_instance_number: {base_instance_number}'

        # 正規化したキーポイント座標とラベルを取得
        norm_keypoints = []
        class_labels = []
        for segment in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
            seg_x = segment + '_nx'
            seg_y = segment + '_ny'
            norm_keypoints.append((target_row[seg_x], target_row[seg_y]))
            class_labels.append(segment)

        if self._use_center:
            # 中心スライスを基準にデータを収集する
            base_index = len(image_paths) // 2
        else:
            # アノテーションされているスライスを基準にデータを収集する
            base_index = base_index

        # 基準スライスを起点にnスライスを選択
        image_paths = self._select_n_elements(image_paths, self._num_slices, base_index)

        # 同一series内に異なる解像度のスライスが含まれるケースがあるため、読み込み時に固定サイズにリサイズする
        images = []
        for image_path in image_paths:
            image = self._read_image(image_path.as_posix())
            images.append(image)
        image = np.stack(images, axis=-1)

        # 正規化キーポイント座標を画像サイズキーポイントに変換
        keypoints = []
        h, w = image.shape[:2]
        for norm_keypoint in norm_keypoints:
            nx, ny = norm_keypoint
            x, y = nx * w, ny * h
            keypoints.append((x, y))

        # データ拡張
        if self._transform is not None:
            transformed = self._transform(image=image, keypoints=keypoints, class_labels=class_labels)
            image = transformed['image']
            keypoints = transformed['keypoints']
            class_labels = transformed['class_labels']

        # ヒートマップのGTを作成
        h, w = image.shape[:2]
        heatmap = np.zeros([len(self._labels), h // self._stride, w // self._stride])
        radius = gaussian_radius([s // self._stride for s in self._heatmap_size])
        radius = max(0, int(radius))
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            kpt_x = keypoint[0] / self._stride
            kpt_y = keypoint[1] / self._stride
            draw_gaussian(heatmap[index], (kpt_x, kpt_y), radius)

        image = image.astype(np.float32)
        if len(image.shape) == 2:
            image = image[None, ...]
        else:
            image = image.transpose(2, 0, 1)

        ret_keypoints = [(-1.0, -1.0) for _ in range(len(self._labels))]
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            ret_keypoints[index] = keypoint
        return image, heatmap, study_id, np.asarray(ret_keypoints)
