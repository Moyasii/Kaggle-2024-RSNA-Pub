# %%
from src.utils import load_settings
from sklearn.model_selection import GroupKFold
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')


# %%
SETTINGS = load_settings()
nfold = 5

# %% [markdown]
# # Initial data inspection

# %%
# Define file paths
train_file_path = SETTINGS.raw_data_dir / "rsna-2024-lumbar-spine-degenerative-classification/train.csv"
train_label_coordinates_path = SETTINGS.raw_data_dir / "rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv"
train_series_descriptions_path = SETTINGS.raw_data_dir / "rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
fold_train_file_path = SETTINGS.train_data_clean_dir / "train_fold.csv"
SETTINGS.train_data_clean_dir.mkdir(exist_ok=True, parents=True)

# Load data
train_data = pd.read_csv(train_file_path)
train_label_coordinates = pd.read_csv(train_label_coordinates_path)
train_series_descriptions = pd.read_csv(train_series_descriptions_path)

# %%
# Merge train_data and train_label_coordinates on 'study_id'
merged_data = pd.merge(train_data, train_label_coordinates, on='study_id', how='left')

# Merge the result with train_series_descriptions on 'study_id' and 'series_id'
merged_data = pd.merge(merged_data, train_series_descriptions, on=['study_id', 'series_id'], how='left')

# %% [markdown]
# # Delete NaN
# %%
merged_data = merged_data[~pd.isnull(merged_data.condition)]

# %% [markdown]
# # Target column

# %%
# 'Spinal Canal Stenosis', 'L1/L2' -> 'spinal_canal_stenosis_l1_l2'


def get_target(row):
    condition = row.condition.lower().replace(' ', '_')
    level = row.level.lower().replace('/', '_')
    col = '_'.join((condition, level))
    return row[col]


# %%
merged_data['target'] = merged_data.apply(get_target, axis=1)

# %%
merged_data = merged_data[~pd.isnull(merged_data['target'])]

# %%
columns = ['study_id', 'series_id', 'instance_number',
           'condition', 'level', 'x', 'y', 'series_description', 'target']
train_df = merged_data[columns]

# %%
train_df['series_id'] = train_df['series_id'].astype(int)
train_df['instance_number'] = train_df['instance_number'].astype(int)

# %% [markdown]
# # Add image path


def get_image_path(row):
    return '/'.join((str(row.study_id), str(row.series_id), str(row.instance_number))) + '.dcm'


train_df['image_path'] = train_df.apply(get_image_path, axis=1)

# %%
del merged_data, train_data, train_label_coordinates, train_series_descriptions
gc.collect()

# %% [markdown]
# # Label encoding

# %%
label_enc = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
train_df['label'] = train_df['target'].map(label_enc)
train_df['bin_label'] = (train_df['label'] > 0).astype(int)

# %% [markdown]
# should be used in validation

# %%
sum_bin_label = train_df.groupby('study_id')['bin_label'].sum().reset_index()
sum_bin_label.rename(columns={'bin_label': 'sum_bin_label'}, inplace=True)
train_df = train_df.merge(sum_bin_label, on='study_id', how='left')


def create_folds(data, n_splits=5):
    data['fold'] = -1

    gkf = GroupKFold(n_splits=n_splits)

    groups = data['study_id']
    stratify_by = data['sum_bin_label']

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, y=stratify_by, groups=groups)):
        data.loc[val_idx, 'fold'] = fold

    return data


train_df = create_folds(train_df, n_splits=nfold)
for fold in range(nfold):
    print(len(train_df[train_df['fold'] == fold]), train_df[train_df['fold'] == fold]['bin_label'].sum())

fold_df = train_df[['study_id', 'fold']].drop_duplicates().reset_index(drop=True)
min_fold = fold_df['fold'].value_counts().idxmin()
fold_df.loc[len(fold_df)] = dict(study_id=3008676218, fold=min_fold)
print(f'add {dict(study_id=3008676218, fold=min_fold)}')

new_train_df = pd.read_csv(train_file_path)
new_train_df = new_train_df.merge(fold_df, on='study_id', how='left')
new_train_df.to_csv(fold_train_file_path, index=False)
