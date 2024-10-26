# %% [markdown]
# # Import Libralies

# %%
import os
import re
import glob
import math
import shutil
import argparse
from pathlib import Path

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
LUMBER_COOD_DATASET_ROOT = SETTINGS.raw_data_dir / 'lumbar_coordinate_dataset'
DST_DIR = SETTINGS.train_data_clean_dir / f'{args.user}/keypoint_datasets/v2'
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
]

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

# %%
LUMBER_DATASET_COOD_DF = pd.read_csv(f'{LUMBER_COOD_DATASET_ROOT}/coords_pretrain.csv')
LUMBER_DATASET_COOD_DF

# %% [markdown]
# ## ラベルを手修正する

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
# 各studyが参照する画像を記録したCSVファイルを作成する。

# %%
data = []
st_ids = DF['study_id'].unique()
descs = ['Sagittal T1', 'Sagittal T2/STIR']

all_segment_studies = []
for idx, study_id in enumerate(tqdm(st_ids, total=len(st_ids))):
    pdf = DF[DF['study_id'] == study_id]

    for desc in descs:
        for series_id in pdf[pdf['series_description'] == desc]['series_id'].values.tolist():
            cood_df = FIXED_COOD_DF[(FIXED_COOD_DF['study_id'] == study_id) & (FIXED_COOD_DF['series_id'] == series_id)].copy()
            if len(cood_df) == 0:
                continue
            dicom_folder = f'{ROOT_DIR}/train_images/{study_id}/{series_id}'
            stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])
            num_slices = len(stack['instance_numbers'])
            lower_idx = math.floor(num_slices * 0.35)
            upper_idx = math.ceil(num_slices * 0.65)
            lower_instance_number = stack['instance_numbers'][lower_idx]
            upper_instance_number = stack['instance_numbers'][upper_idx]
            cood_df = cood_df[(cood_df['instance_number'] >= lower_instance_number) & (cood_df['instance_number'] <= upper_instance_number)]
            count_df = cood_df[['instance_number', 'condition']].value_counts().reset_index()
            count_df = count_df[count_df['count'] == 5]

            for i, count_row in count_df.iterrows():
                instance_number = count_row.instance_number
                condition = count_row.condition

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
                    condition=condition,
                    standard_pos=standard_pos,
                    png_path=png_path.relative_to(DST_DIR),
                )

                instance_cood_df = cood_df[cood_df['instance_number'] == instance_number].copy()
                for i, cood_row in instance_cood_df.iterrows():
                    segment = cood_row.level
                    anno[segment + '_x'] = cood_row.x
                    anno[segment + '_y'] = cood_row.y

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
train_df.to_csv(DST_DIR / 'train_5folds.csv', index=False)

# %% [markdown]
# ## Lumber Coordinate Datasetの追加

# %% [markdown]
# filename	source	x	y	level	relative_x	relative_y
# 1_t2.jpg	spider	139	175	L5/S1	0.542969	0.683594

# %%
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
    dst_file_path = DST_DIR / 'lumber_images_jpg' / f'processed_{source}_jpgs' / filename
    if not dst_file_path.parent.exists():
        dst_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file_path, dst_file_path)

    anno = dict(
        source=source,
        study_id=10000000000 + i,
        series_id=-1,
        desc=desc_table[source],
        instance_number=-1,
        condition='unknown',
        standard_pos=-1,
        png_path=dst_file_path.relative_to(DST_DIR),
    )
    for segment in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
        anno[segment + '_x'] = cood_df.loc[cood_df['level'] == segment, 'x'].values[0]
        anno[segment + '_y'] = cood_df.loc[cood_df['level'] == segment, 'y'].values[0]

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
pd.concat([train_df, train_lumber_df]).to_csv(DST_DIR / 'train_all_5folds.csv', index=False)
