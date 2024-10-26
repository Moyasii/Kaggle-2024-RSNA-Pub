# %% [markdown]
# # Import Libralies

# %%

import os
import re
import glob
import argparse

import pydicom
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold

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
DST_DIR = SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v5'
DST_DIR.mkdir(exist_ok=True, parents=True)

NUM_FOLDS = 10
SEED = args.seed

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


def select_n_elements(lst, n):
    # リストの長さを取得
    length = len(lst)

    # nがリストの長さ以上の場合は、そのままリストを返す
    if n >= length:
        return lst

    # 等間隔で選ぶためのインデックスを計算
    indices = np.linspace(0, length - 1, n).astype(int)

    # インデックスを基にリストから要素を取得
    result = np.array(lst)[indices]

    return result

# %% [markdown]
# # Reading and Taking a look csv


# %%
DF = pd.read_csv(f'{ROOT_DIR}/train_series_descriptions.csv')
DF.head()

# %%
COOD_DF = pd.read_csv(f'{ROOT_DIR}/train_label_coordinates.csv')
COOD_DF.head()

# %% [markdown]
# ## 椎間関節下狭窄が左右両方にアノテされている画像を抽出する

# %%
LR_SUBARTICULAR_STENOSIS_DF = COOD_DF.groupby(['study_id', 'series_id', 'instance_number']).filter(
    lambda x: set(x['condition']) == {"Right Subarticular Stenosis", "Left Subarticular Stenosis"}
)

# %% [markdown]
# ## 脊柱管位置を作成する
#
# 左右狭窄位置の中間点を脊柱管位置とする。

# %%
data = []
for i, (_, row) in enumerate(LR_SUBARTICULAR_STENOSIS_DF[['study_id', 'series_id', 'instance_number']].drop_duplicates().iterrows()):
    subset_df = LR_SUBARTICULAR_STENOSIS_DF[(LR_SUBARTICULAR_STENOSIS_DF['study_id'] == row.study_id) &
                                            (LR_SUBARTICULAR_STENOSIS_DF['series_id'] == row.series_id) &
                                            (LR_SUBARTICULAR_STENOSIS_DF['instance_number'] == row.instance_number)]
    assert len(subset_df) == 2
    right_row = subset_df[subset_df['condition'] == 'Right Subarticular Stenosis'].iloc[0]
    left_row = subset_df[subset_df['condition'] == 'Left Subarticular Stenosis'].iloc[0]
    mid_x = (left_row['x'] + right_row['x']) / 2
    mid_y = (left_row['y'] + right_row['y']) / 2
    data.append({
        'study_id': row.study_id,
        'series_id': row.series_id,
        'instance_number': row.instance_number,
        'left_level': left_row['level'],
        'right_level': right_row['level'],
        'x': mid_x,
        'y': mid_y,
    })
SPINAL_CANAL_DF = pd.DataFrame(data)
SPINAL_CANAL_DF

# %% [markdown]
# ## データセットを作成
#
# データセットを作成する

# %%
data = []
study_ids = SPINAL_CANAL_DF['study_id'].unique()

for i, study_id in enumerate(tqdm(study_ids, total=len(study_ids))):
    subset_df = SPINAL_CANAL_DF[SPINAL_CANAL_DF['study_id'] == study_id]
    series_ids = subset_df['series_id'].unique()
    for series_id in series_ids:
        target_df = subset_df[subset_df['series_id'] == series_id]
        desc = DF[(DF['study_id'] == study_id) & (DF['series_id'] == series_id)]['series_description'].values[0]
        dicom_folder = f'{ROOT_DIR}/train_images/{study_id}/{series_id}'
        stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])
        for _, row in target_df.iterrows():
            instance_number = row.instance_number

            index = np.where(stack['instance_numbers'] == instance_number)[0][0]
            image = apply_min_max_norm(stack['arrays'][index])
            standard_pos = stack['standard_positions'][index]
            png_path = DST_DIR / f'train_images_png/{study_id}/{series_id}/{instance_number}.png'
            if not png_path.exists():
                png_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(png_path, image)

            anno = dict(
                source='rsna',
                study_id=study_id,
                series_id=series_id,
                desc=desc,
                instance_number=instance_number,
                condition='N/A',
                standard_pos=standard_pos,
                png_path=png_path.relative_to(DST_DIR),
                x=row.x,
                y=row.y,
            )
            data.append(anno)

train_df = pd.DataFrame(data=data)
train_df = train_df.drop_duplicates()
train_df.to_csv(DST_DIR / 'train.csv', index=False)

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
train_df.to_csv(DST_DIR / f'train_{NUM_FOLDS}folds.csv', index=False)
