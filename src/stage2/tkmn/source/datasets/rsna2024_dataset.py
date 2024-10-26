# reference: https://www.kaggle.com/code/hugowjd/rsna2024-lsdc-training-densenet#Define-Dataset

from typing import Optional

import copy
import math

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations as A

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.datasets.dataset_phase import DatasetPhase  # noqa
from source.utils.labels import CONDITION_TO_ID, LEVEL_TO_ID  # noqa


def read_image(image_path: str,
               crop_h: Optional[tuple[float, float]] = None,
               crop_w: Optional[tuple[float, float]] = None,
               image_size: Optional[tuple[int, int]] = None,
               ) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if crop_h is not None:
        h, w = image.shape
        start_h = int(h * crop_h[0])
        end_h = int(h * crop_h[1])
        image = image[start_h:end_h, :]
    if crop_w is not None:
        h, w = image.shape
        start_w = int(w * crop_w[0])
        end_w = int(w * crop_w[1])
        image = image[:, start_w:end_w]
    if image_size is not None:
        image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8)


class RSNA2024DatasetVer2(Dataset):
    # | #   | サジタルT1 | サジタルT2 | アキシャルT2 | レベル分割 | LR分割 | 中心抽出 | 出力                         | 備考                                                                                 |
    # | --- | ---------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------------------- | ------------------------------------------------------------------------------------ |
    # | 2   | ○          | ○          | ○            | ○          |        |          | 15クラス (condition×3)       | レベル分割データから対応するレベルの狭窄を推定する                                   |
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        train_image_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        target_conditions: list[str] = ['spinal_canal_stenosis',
                                        'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
                                        'left_subarticular_stenosis', 'right_subarticular_stenosis'],
        target_levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        transforms: Optional[dict[A.Compose]] = {
            'Sagittal T1': None,
            'Sagittal T2/STIR': None,
            'Axial T2': None,
        },
        image_shapes: dict[str, tuple[int, int, int]] = {
            'Sagittal T1': (512, 512, 10),
            'Sagittal T2/STIR': (512, 512, 10),
            'Axial T2': (512, 512, 10),
        }
    ) -> None:
        self.level_to_id = copy.deepcopy(LEVEL_TO_ID)
        self.condition_to_id = copy.deepcopy(CONDITION_TO_ID)
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        if 'fold' in self._train_df.columns:
            self._train_df = self._train_df.drop(columns=['fold'])
        self._train_image_df = train_image_df.copy()

        # 疾患を抽出
        self._target_conditions = target_conditions
        self._num_conditions = len(target_conditions)
        target_columns = ['study_id']
        for col in self._train_df.columns[1:]:
            if any(condition in col for condition in self._target_conditions):
                target_columns.append(col)
        self._train_df = self._train_df[target_columns]

        # 椎間板レベルを抽出
        self.target_levels = target_levels
        new_train_df_list = []
        for i, level in enumerate(self.level_to_id.keys()):
            if level in self.target_levels:
                level_label_df = self._train_df.iloc[:, [0] + list(range(1 + i, self._train_df.shape[1], self._num_conditions))].copy().reset_index(drop=True)
                level_label_df.columns = ['study_id'] + target_conditions
                level_label_df.insert(1, 'level', level)
                new_train_df_list.append(level_label_df)
        self._train_df = pd.concat(new_train_df_list).sort_values(['study_id', 'level']).reset_index(drop=True)

        self._phase = phase
        self._transforms = transforms
        self._image_shapes = image_shapes
        self._cache = {}

    def __len__(self):
        return len(self._train_df)

    def _select_n_elements(self, lst: np.ndarray, n: int) -> np.ndarray:
        # リストの長さを取得
        length = len(lst)
        if length == 0:
            return lst

        # 等間隔で選ぶためのインデックスを計算（四捨五入を使用）
        indices = np.round(np.linspace(0, length - 1, n)).astype(int)

        # インデックスを基にリストから要素を取得
        result = lst[indices]
        return result

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        level = target_row.iloc[1]
        label = target_row.iloc[2:].values.astype(np.int64).reshape(self._num_conditions, 1)
        study_id = int(target_row['study_id'])
        study_image_df = self._train_image_df[(self._train_image_df['study_id'] == study_id) & (self._train_image_df['level'] == level)]
        available_flag = True

        plane_images = {}
        no_image_planes = []
        for plane in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            if plane == 'Axial T2':
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            else:
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)
            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    plane_image[..., i] = read_image((self._image_root / image_path).as_posix(), image_size=image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        if len(no_image_planes) == 3:
            # すべて画像が存在しないときはラベルをすべて-100にする（ロス算出時に無視される）
            label = np.array([-100] * self._num_conditions, dtype=np.int64).reshape(self._num_conditions, 1)
            available_flag = False

        condition_id = np.array([self.condition_to_id[c] for c in self._target_conditions])
        level_id = np.array([self.level_to_id[level]])

        return plane_images['Sagittal T1'], plane_images['Sagittal T2/STIR'], plane_images['Axial T2'], label, study_id, level_id, condition_id, available_flag


class RSNA2024DatasetVer5(Dataset):
    # | #   | サジタルT1 | サジタルT2 | アキシャルT2 | レベル分割 | LR分割 | 中心抽出 | 出力                         | 備考                                                                                 |
    # | --- | ---------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------------------- | ------------------------------------------------------------------------------------ |
    # | 5   | ○          | ○          | ○            | ○          | ○      |          | 6クラス (2×3)                | レベル分割・左右分割データから対応するレベル・サイドの椎間孔・椎間関節狭窄を推定する |
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        train_image_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        target_conditions: list[str] = ['neural_foraminal_narrowing', 'subarticular_stenosis'],
        target_levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        target_sides: list[str] = ['left', 'right'],
        transforms: Optional[dict[A.Compose]] = {
            'Sagittal T1': None,
            'Sagittal T2/STIR': None,
            'Axial T2': None,
        },
        image_shapes: dict[str, tuple[int, int, int]] = {
            'Sagittal T1': (512, 512, 10),
            'Sagittal T2/STIR': (512, 512, 10),
            'Axial T2': (512, 512, 10),
        }
    ) -> None:
        self.level_to_id = copy.deepcopy(LEVEL_TO_ID)
        self.condition_to_id = copy.deepcopy(CONDITION_TO_ID)
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        if 'fold' in self._train_df.columns:
            self._train_df = self._train_df.drop(columns=['fold'])
        self._train_image_df = train_image_df.copy()

        # 疾患を抽出
        self._target_conditions = target_conditions
        self._num_conditions = len(target_conditions)
        target_columns = ['study_id']
        for col in self._train_df.columns[1:]:
            if any(condition in col for condition in self._target_conditions):
                target_columns.append(col)
        self._train_df = self._train_df[target_columns]

        # 左右を抽出
        self.target_sides = target_sides
        new_train_df_list = []
        for i, side in enumerate(['left', 'right']):
            if side in self.target_sides:
                target_columns = ['study_id'] + [col for col in self._train_df.columns if side in col]
                side_label_df = self._train_df[target_columns].copy().reset_index(drop=True)
                new_columns = [col.replace(f'{side}_', '') for col in target_columns]
                side_label_df.columns = new_columns
                side_label_df.insert(1, 'side', side)
                new_train_df_list.append(side_label_df)
        self._train_df = pd.concat(new_train_df_list).sort_values(['study_id', 'side']).reset_index(drop=True)

        # 椎間板レベルを抽出
        self.target_levels = target_levels
        new_train_df_list = []
        for i, level in enumerate(self.level_to_id.keys()):
            if level in self.target_levels:
                _level = level.replace('/', '_').lower()
                target_columns = ['study_id', 'side'] + [col for col in self._train_df.columns if _level in col]
                level_label_df = self._train_df[target_columns].copy().reset_index(drop=True)
                new_columns = [col.replace(f'_{_level}', '') for col in target_columns]
                level_label_df.columns = new_columns
                level_label_df.insert(1, 'level', level)
                new_train_df_list.append(level_label_df)
        self._train_df = pd.concat(new_train_df_list).sort_values(['study_id', 'level']).reset_index(drop=True)

        self._phase = phase
        self._transforms = transforms
        self._image_shapes = image_shapes
        self._cache = {}

    def __len__(self):
        return len(self._train_df)

    def _select_n_elements(self, lst: np.ndarray, n: int) -> np.ndarray:
        # リストの長さを取得
        length = len(lst)
        if length == 0:
            return lst

        # 等間隔で選ぶためのインデックスを計算（四捨五入を使用）
        indices = np.round(np.linspace(0, length - 1, n)).astype(int)

        # インデックスを基にリストから要素を取得
        result = lst[indices]

        return result

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        level = target_row.iloc[1]
        side = target_row.iloc[2]
        label = target_row.iloc[3:].values.astype(np.int64).reshape(self._num_conditions, 1)
        study_id = int(target_row['study_id'])
        study_image_df = self._train_image_df[(self._train_image_df['study_id'] == study_id) & (self._train_image_df['level'] == level)]
        available_flag = True

        plane_images = {}
        no_image_planes = []
        for plane in ['Sagittal T1', 'Sagittal T2/STIR']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            if side == 'right':
                # 昇順（右→左）
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)
            else:
                # 降順（左→右）
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            # 体の右端・左端から中心までのスライスを取得する
            plane_study_image_df = plane_study_image_df.iloc[:math.ceil(len(plane_study_image_df)/2)]

            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    plane_image[..., i] = read_image((self._image_root / image_path).as_posix(),
                                                     image_size=image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        for plane in ['Axial T2']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    if side == 'right':
                        # 画像の左側が体の右側
                        image = read_image((self._image_root / image_path).as_posix(),
                                           crop_w=(0.0, 0.6), image_size=image_shape[:2])
                        image = cv2.flip(image, 1)  # 向きをそろえる。
                    else:
                        image = read_image((self._image_root / image_path).as_posix(),
                                           crop_w=(0.4, 1.0), image_size=image_shape[:2])
                    plane_image[..., i] = image
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        if len(no_image_planes) == 3:
            # すべて画像が存在しないときはラベルをすべて-100にする（ロス算出時に無視される）
            label = np.array([-100] * self._num_conditions, dtype=np.int64).reshape(self._num_conditions, 1)
            available_flag = False

        condition_id = np.array([self.condition_to_id[f'{side}_{c}'] for c in self._target_conditions])
        level_id = np.array([self.level_to_id[level]])

        return plane_images['Sagittal T1'], plane_images['Sagittal T2/STIR'], plane_images['Axial T2'], label, study_id, level_id, condition_id, available_flag


class RSNA2024DatasetVer6(Dataset):
    # | #   | サジタルT1 | サジタルT2 | アキシャルT2 | レベル分割 | LR分割 | 中心抽出 | 出力                         | 備考                                                                                 |
    # | --- | ---------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------------------- | ------------------------------------------------------------------------------------ |
    # | 6   | ○          | ○          | ○            | ○          |        | ○        | 3クラス (1×3)                | レベル分割・中央抽出データから対応するレベルの脊柱管狭窄を推定する                   |
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        train_image_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        target_conditions: list[str] = ['spinal_canal_stenosis'],
        target_levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        sagittal_start_pos: float = 0.25,
        sagittal_end_pos: float = 0.75,
        transforms: Optional[dict[A.Compose]] = {
            'Sagittal T1': None,
            'Sagittal T2/STIR': None,
            'Axial T2': None,
        },
        image_shapes: dict[str, tuple[int, int, int]] = {
            'Sagittal T1': (512, 512, 10),
            'Sagittal T2/STIR': (512, 512, 10),
            'Axial T2': (512, 512, 10),
        },
        h_flip: float = 0.0,
    ) -> None:
        self.level_to_id = copy.deepcopy(LEVEL_TO_ID)
        self.condition_to_id = copy.deepcopy(CONDITION_TO_ID)
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        if 'fold' in self._train_df.columns:
            self._train_df = self._train_df.drop(columns=['fold'])
        self._train_image_df = train_image_df.copy()

        # 疾患を抽出
        self._target_conditions = target_conditions
        self._num_conditions = len(target_conditions)
        target_columns = ['study_id']
        for col in self._train_df.columns[1:]:
            if any(condition in col for condition in self._target_conditions):
                target_columns.append(col)
        self._train_df = self._train_df[target_columns]

        # 椎間板レベルを抽出
        self.target_levels = target_levels
        new_train_df_list = []
        for i, level in enumerate(self.level_to_id.keys()):
            if level in self.target_levels:
                _level = level.replace('/', '_').lower()
                target_columns = ['study_id'] + [col for col in self._train_df.columns if _level in col]
                level_label_df = self._train_df[target_columns].copy().reset_index(drop=True)
                new_columns = [col.replace(f'_{_level}', '') for col in target_columns]
                level_label_df.columns = new_columns
                level_label_df.insert(1, 'level', level)
                new_train_df_list.append(level_label_df)
        self._train_df = pd.concat(new_train_df_list).sort_values(['study_id', 'level']).reset_index(drop=True)

        self._sagittal_start_pos = sagittal_start_pos
        self._sagittal_end_pos = sagittal_end_pos
        self._phase = phase
        self._transforms = transforms
        self._image_shapes = image_shapes
        self._h_flip = h_flip if phase == DatasetPhase.TRAIN else 0.0
        self._cache = {}

    def __len__(self):
        return len(self._train_df)

    def _select_n_elements(self, lst: np.ndarray, n: int) -> np.ndarray:
        # リストの長さを取得
        length = len(lst)
        if length == 0:
            return lst

        # 等間隔で選ぶためのインデックスを計算（四捨五入を使用）
        indices = np.round(np.linspace(0, length - 1, n)).astype(int)

        # インデックスを基にリストから要素を取得
        result = lst[indices]

        return result

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        level = target_row.iloc[1]
        label = target_row.iloc[2:].values.astype(np.int64).reshape(self._num_conditions, 1)
        study_id = int(target_row['study_id'])
        study_image_df = self._train_image_df[(self._train_image_df['study_id'] == study_id) & (self._train_image_df['level'] == level)]
        available_flag = True
        is_h_flip = np.random.rand() < self._h_flip

        plane_images = {}
        no_image_planes = []
        for plane in ['Sagittal T1', 'Sagittal T2/STIR']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            if is_h_flip:
                # X軸方向に反転（左→右）
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            else:
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)

            # 体の中心近傍のスライスを取得する
            start_idx = math.floor(len(plane_study_image_df) * self._sagittal_start_pos)
            end_idx = math.ceil(len(plane_study_image_df) * self._sagittal_end_pos)
            plane_study_image_df = plane_study_image_df.iloc[start_idx:end_idx]

            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    plane_image[..., i] = read_image((self._image_root / image_path).as_posix(),
                                                     image_size=image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        for plane in ['Axial T2']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    image = read_image((self._image_root / image_path).as_posix(), image_size=image_shape[:2])
                    if is_h_flip:
                        # X軸方向に反転（左→右）
                        image = cv2.flip(image, 1)
                    plane_image[..., i] = image
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        if len(no_image_planes) == 3:
            # すべて画像が存在しないときはラベルをすべて-100にする（ロス算出時に無視される）
            label = np.array([-100] * self._num_conditions, dtype=np.int64).reshape(self._num_conditions, 1)
            available_flag = False

        condition_id = np.array([self.condition_to_id[c] for c in self._target_conditions])
        level_id = np.array([self.level_to_id[level]])

        return plane_images['Sagittal T1'], plane_images['Sagittal T2/STIR'], plane_images['Axial T2'], label, study_id, level_id, condition_id, available_flag


class RSNA2024DatasetVer7(Dataset):
    # | #   | サジタルT1 | サジタルT2 | アキシャルT2 | レベル分割 | LR分割 | 中心抽出 | 出力                         | 備考                                                                                 |
    # | --- | ---------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------------------- | ------------------------------------------------------------------------------------ |
    # | 7   | ○          | ○          | ○            | ○          |        |          | 75クラス (condition×level×3) | レベル分割データをからすべてのレベル・サイドの狭窄を推定する                         |
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        train_image_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        target_conditions: list[str] = ['spinal_canal_stenosis',
                                        'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
                                        'left_subarticular_stenosis', 'right_subarticular_stenosis'],
        transforms: Optional[dict[A.Compose]] = {
            'Sagittal T1': None,
            'Sagittal T2/STIR': None,
            'Axial T2': None,
        },
        image_shapes: dict[str, tuple[int, int, int]] = {
            'Sagittal T1': (512, 512, 10),
            'Sagittal T2/STIR': (512, 512, 10),
            'Axial T2': (512, 512, 10),
        }
    ) -> None:
        self.level_to_id = copy.deepcopy(LEVEL_TO_ID)
        self.condition_to_id = copy.deepcopy(CONDITION_TO_ID)
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        if 'fold' in self._train_df.columns:
            self._train_df = self._train_df.drop(columns=['fold'])
        self._train_image_df = train_image_df.copy()

        # 疾患を抽出
        self._target_conditions = target_conditions
        self._num_conditions = len(target_conditions)
        target_columns = ['study_id']
        for col in self._train_df.columns[1:]:
            if any(condition in col for condition in self._target_conditions):
                target_columns.append(col)
        self._train_df = self._train_df[target_columns]

        self._phase = phase
        self._transforms = transforms
        self._image_shapes = image_shapes

    def __len__(self):
        return len(self._train_df)

    def _select_n_elements(self, lst: np.ndarray, n: int) -> np.ndarray:
        # リストの長さを取得
        length = len(lst)
        if length == 0:
            return lst

        # 等間隔で選ぶためのインデックスを計算（四捨五入を使用）
        indices = np.round(np.linspace(0, length - 1, n)).astype(int)

        # インデックスを基にリストから要素を取得
        result = lst[indices]
        return result

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        label = target_row.iloc[1:].values.astype(np.int64).reshape(self._num_conditions, 5)
        study_id = int(target_row['study_id'])
        study_image_df = self._train_image_df[(self._train_image_df['study_id'] == study_id)]
        available_flag = True

        plane_images = {}
        no_image_planes = []
        for plane in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
            lavel_images = []
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane]
            for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
                image_shape = self._image_shapes[plane]
                transform = self._transforms[plane]
                level_image = np.zeros(image_shape, dtype=np.uint8)
                level_channel = image_shape[-1]
                level_study_image_df = plane_study_image_df[plane_study_image_df['level'] == level].copy()

                # 複数のシリーズが存在する場合はランダムに選択
                if len(level_study_image_df['series_id'].unique()) > 1:
                    selected_series_id = np.random.choice(level_study_image_df['series_id'].unique())
                    level_study_image_df = level_study_image_df[level_study_image_df['series_id'] == selected_series_id]

                if plane == 'Axial T2':
                    level_study_image_df = level_study_image_df.sort_values('standard_pos', ascending=False)
                else:
                    level_study_image_df = level_study_image_df.sort_values('standard_pos', ascending=True)
                image_paths = self._select_n_elements(level_study_image_df['image_path'].values, level_channel)
                if len(image_paths) > 0:
                    for i, image_path in enumerate(image_paths):
                        level_image[..., i] = read_image((self._image_root / image_path).as_posix(), image_size=image_shape[:2])
                else:
                    no_image_planes.append(f'{plane}_{level}')
                lavel_images.append(level_image)

            plane_image = np.concatenate(lavel_images, axis=-1)  # (height, width, level * channel)
            if transform is not None:
                plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        if len(no_image_planes) == 15:
            # すべて画像が存在しないときはラベルをすべて-100にする（ロス算出時に無視される）
            label = np.array([-100] * self._num_conditions * 5, dtype=np.int64).reshape(self._num_conditions, 5)
            available_flag = False

        condition_id = np.array([self.condition_to_id[c] for c in self._target_conditions])
        level_id = np.array([self.level_to_id[level] for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']])

        return plane_images['Sagittal T1'], plane_images['Sagittal T2/STIR'], plane_images['Axial T2'], label, study_id, level_id, condition_id, available_flag


class RSNA2024DatasetVer8(Dataset):
    # | #   | サジタルT1 | サジタルT2 | アキシャルT2 | レベル分割 | LR分割 | 中心抽出 | 出力                         | 備考                                                                                 |
    # | --- | ---------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------------------- | ------------------------------------------------------------------------------------ |
    # | 8   |            | ○          | ○            | ○          |        | ○        | 3クラス (1×3)                | レベル分割・中央抽出データから対応するレベルの脊柱管狭窄を推定する                   |
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        train_image_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        target_conditions: list[str] = ['spinal_canal_stenosis'],
        target_levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        sagittal_start_pos: float = 0.25,
        sagittal_end_pos: float = 0.75,
        transforms: Optional[dict[A.Compose]] = {
            'Sagittal T2/STIR': None,
            'Axial T2': None,
        },
        image_shapes: dict[str, tuple[int, int, int]] = {
            'Sagittal T2/STIR': (512, 512, 10),
            'Axial T2': (512, 512, 10),
        },
        h_flip: float = 0.0,
    ) -> None:
        self.level_to_id = copy.deepcopy(LEVEL_TO_ID)
        self.condition_to_id = copy.deepcopy(CONDITION_TO_ID)
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        if 'fold' in self._train_df.columns:
            self._train_df = self._train_df.drop(columns=['fold'])
        self._train_image_df = train_image_df.copy()

        # 疾患を抽出
        self._target_conditions = target_conditions
        self._num_conditions = len(target_conditions)
        target_columns = ['study_id']
        for col in self._train_df.columns[1:]:
            if any(condition in col for condition in self._target_conditions):
                target_columns.append(col)
        self._train_df = self._train_df[target_columns]

        # 椎間板レベルを抽出
        self.target_levels = target_levels
        new_train_df_list = []
        for i, level in enumerate(self.level_to_id.keys()):
            if level in self.target_levels:
                _level = level.replace('/', '_').lower()
                target_columns = ['study_id'] + [col for col in self._train_df.columns if _level in col]
                level_label_df = self._train_df[target_columns].copy().reset_index(drop=True)
                new_columns = [col.replace(f'_{_level}', '') for col in target_columns]
                level_label_df.columns = new_columns
                level_label_df.insert(1, 'level', level)
                new_train_df_list.append(level_label_df)
        self._train_df = pd.concat(new_train_df_list).sort_values(['study_id', 'level']).reset_index(drop=True)

        self._sagittal_start_pos = sagittal_start_pos
        self._sagittal_end_pos = sagittal_end_pos
        self._phase = phase
        self._transforms = transforms
        self._image_shapes = image_shapes
        self._h_flip = h_flip if phase == DatasetPhase.TRAIN else 0.0
        self._cache = {}

    def __len__(self):
        return len(self._train_df)

    def _select_n_elements(self, lst: np.ndarray, n: int) -> np.ndarray:
        # リストの長さを取得
        length = len(lst)
        if length == 0:
            return lst

        # 等間隔で選ぶためのインデックスを計算（四捨五入を使用）
        indices = np.round(np.linspace(0, length - 1, n)).astype(int)

        # インデックスを基にリストから要素を取得
        result = lst[indices]

        return result

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        level = target_row.iloc[1]
        label = target_row.iloc[2:].values.astype(np.int64).reshape(self._num_conditions, 1)
        study_id = int(target_row['study_id'])
        study_image_df = self._train_image_df[(self._train_image_df['study_id'] == study_id) & (self._train_image_df['level'] == level)]
        available_flag = True
        is_h_flip = np.random.rand() < self._h_flip

        plane_images = {}
        no_image_planes = []
        for plane in ['Sagittal T2/STIR']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            if is_h_flip:
                # X軸方向に反転（左→右）
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            else:
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)

            # 体の中心近傍のスライスを取得する
            start_idx = math.floor(len(plane_study_image_df) * self._sagittal_start_pos)
            end_idx = math.ceil(len(plane_study_image_df) * self._sagittal_end_pos)
            plane_study_image_df = plane_study_image_df.iloc[start_idx:end_idx]

            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    plane_image[..., i] = read_image((self._image_root / image_path).as_posix(),
                                                     image_size=image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        for plane in ['Axial T2']:
            image_shape = self._image_shapes[plane]
            transform = self._transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            image_paths = self._select_n_elements(plane_study_image_df['image_path'].values, plane_channel)
            if len(image_paths) > 0:
                for i, image_path in enumerate(image_paths):
                    image = read_image((self._image_root / image_path).as_posix(), image_size=image_shape[:2])
                    if is_h_flip:
                        # X軸方向に反転（左→右）
                        image = cv2.flip(image, 1)
                    plane_image[..., i] = image
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            else:
                no_image_planes.append(plane)
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        if len(no_image_planes) == 3:
            # すべて画像が存在しないときはラベルをすべて-100にする（ロス算出時に無視される）
            label = np.array([-100] * self._num_conditions, dtype=np.int64).reshape(self._num_conditions, 1)
            available_flag = False

        condition_id = np.array([self.condition_to_id[c] for c in self._target_conditions])
        level_id = np.array([self.level_to_id[level]])

        return torch.tensor(0), plane_images['Sagittal T2/STIR'], plane_images['Axial T2'], label, study_id, level_id, condition_id, available_flag


if __name__ == '__main__':
    from source.datasets.rsna2024_transforms import build_transforms

    image_root = '/kaggle/explorations/create_datasets/create_dataset_for_classificastion_ver4'

    # train
    train_df = pd.read_csv('/kaggle/input/team/train_team_folds.csv')
    train_df = train_df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    train_df = train_df.replace(label2id)
    train_image_df = pd.read_csv('/kaggle/explorations/create_datasets/create_dataset_for_classificastion_ver4/train_images.csv')
    transform = build_transforms(phase=DatasetPhase.TRAIN, image_size=(224, 224))

    train_dataset = RSNA2024DatasetVer2(image_root, train_df, train_image_df,
                                        image_shapes={  # 画像のサイズとスライス数
                                            'Sagittal T1': (224, 224, 15),
                                            'Sagittal T2/STIR': (224, 224, 15),
                                            'Axial T2': (224, 224, 10),
                                        },
                                        transforms={
                                            'Sagittal T1': transform,
                                            'Sagittal T2/STIR': transform,
                                            'Axial T2': transform,
                                        })
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_id, level_ld, condition_id, available_flag = batch
        break

    train_dataset = RSNA2024DatasetVer5(image_root, train_df, train_image_df,
                                        image_shapes={  # 画像のサイズとスライス数
                                            'Sagittal T1': (224, 224, 15),
                                            'Sagittal T2/STIR': (224, 224, 15),
                                            'Axial T2': (224, 224, 10),
                                        },
                                        transforms={
                                            'Sagittal T1': transform,
                                            'Sagittal T2/STIR': transform,
                                            'Axial T2': transform,
                                        })
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_id, level_ld, condition_id, available_flag = batch
        break

    train_dataset = RSNA2024DatasetVer6(image_root, train_df, train_image_df,
                                        image_shapes={  # 画像のサイズとスライス数
                                            'Sagittal T1': (224, 224, 15),
                                            'Sagittal T2/STIR': (224, 224, 15),
                                            'Axial T2': (224, 224, 10),
                                        },
                                        transforms={
                                            'Sagittal T1': transform,
                                            'Sagittal T2/STIR': transform,
                                            'Axial T2': transform,
                                        },
                                        h_flip=0.5)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_id, level_ld, condition_id, available_flag = batch
        break

    train_dataset = RSNA2024DatasetVer7(image_root, train_df, train_image_df,
                                        image_shapes={  # 画像のサイズとスライス数
                                            'Sagittal T1': (224, 224, 15),
                                            'Sagittal T2/STIR': (224, 224, 15),
                                            'Axial T2': (224, 224, 10),
                                        },
                                        transforms={
                                            'Sagittal T1': transform,
                                            'Sagittal T2/STIR': transform,
                                            'Axial T2': transform,
                                        })
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_id, level_ld, condition_id, available_flag = batch
        break

    train_dataset = RSNA2024DatasetVer8(image_root, train_df, train_image_df,
                                        image_shapes={  # 画像のサイズとスライス数
                                            'Sagittal T2/STIR': (224, 224, 15),
                                            'Axial T2': (224, 224, 10),
                                        },
                                        transforms={
                                            'Sagittal T2/STIR': transform,
                                            'Axial T2': transform,
                                        })
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_id, level_ld, condition_id, available_flag = batch
        break
