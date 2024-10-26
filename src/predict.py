# %% [markdown]
# Best CV: 0.3643366894165819
# Spinal:
#     - moyashii_exp0113_脊柱管モデル(cntiny)_224
#     - moyashii_exp0152_(effv2s)_224
#     - moyashii_exp0176_脊柱管強化モデルv9(cntiny)_224
#     - moyashii_exp0188_脊柱管強化モデルv9(cntiny)_224
#     - moyashii_exp0196_脊柱管強化モデルv9(effb4)_224
#     - moyashii_exp0211_脊柱管強化モデルv9(cnnano)_320
#     - tkmn_spinal_cfg0123_aug04_convnext_tiny_cv012340.3994
#     - tkmn_spinal_cfg0141_aug02_efficientnetv2_rw_t_cv012340.3980
#     - tkmn_spinal_cfg0143_aug04_efficientnetv2_rw_t_cv012340.3998
#     - tkmn_spinal_cfg0162_cw1_1_convnext_nano_cv012340.3968
#     - tkmn_spinal_cfg0188_bs2_convnext_tiny_cv012340.3989
#     - tkmn_spinal_cfg0189_bs3_convnext_tiny_cv012340.3970
#     - tkmn_spinal_cfg0202_bs3_mnasnet_small_cv012340.4033
#     - tkmn_spinal_cfg0204_bs2_resnet18_cv012340.4014
#     - tkmn_spinal_cfg0211_bs3_efficientnetv2_rw_t_cv012340.3979
# LR:
#     - moyashii_exp0082_LR(cnnano)_224
#     - moyashii_exp0157_LRモデルv9(effv2s)_224
#     - moyashii_exp0159_LRモデルv9(cntiny)_224
#     - moyashii_exp0177_LRモデルv9(cntiny)_224
#     - moyashii_exp0191_LRモデルv9(cntiny)_224
#     - moyashii_exp0200_LRモデルv9(cnnano)_224
#     - moyashii_exp0212_LRモデルv9(cnnano)_320
#     - suguuuuu_exp2064_base-conv-batch2_fold_0.40268_submission
#     - tkmn_cfg0101_aug00_convnext_nano_cv012340.4044
#     - tkmn_cfg0123_aug02_r160_resnet18_cv012340.4045
#     - tkmn_cfg0124_aug03_r160_resnet18_cv012340.4040
#     - tkmn_cfg0138_aug02_r256_maxvit_rmlp_nano_rw_256_cv012340.4032
#     - tkmn_cfg0139_aug03_r256_maxvit_rmlp_nano_rw_256_cv012340.4018
#     - tkmn_cfg0145_aug04_r256_maxvit_nano_rw_256_cv012340.4047
#     - tkmn_cfg0203_bs3_mnasnet_small_cv012340.4062


# %% [markdown]
# ## Import Libralies

# %%
from typing import Optional

import os
import re
import glob
import math
import argparse

import cv2
import pydicom
import numpy as np
from scipy.interpolate import UnivariateSpline

import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import albumentations as A

from omegaconf import OmegaConf

from src.utils import load_settings

parser = argparse.ArgumentParser()
parser.add_argument('--use_pretrain', action='store_true')
args = parser.parse_args()

# %% [markdown]
# ## Config
SETTINGS = load_settings()

IMAGE_ROOT = SETTINGS.raw_data_dir / 'rsna-2024-lumbar-spine-degenerative-classification/test_images'
TEST_DESC_CSV = SETTINGS.raw_data_dir / 'rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv'
TEST_DESC_DF = pd.read_csv(TEST_DESC_CSV)

# SUBMISSION
SUBMISSION_CSV = SETTINGS.submission_dir / 'submission.csv'

# GPU
DEVICE0 = torch.device('cuda:0')

# DATASET (v8)
DATASET_V8_SAGITTAL_CROP_MM = [40, 60]  # h:35mm, w:50mm
DATASET_V8_SAGITTAL_CROP_OFFSET_MM = [0, 0]  # h:0mm, w:0mm
DATASET_V8_AXIAL_CROP_MM = [80, 70]  # h:45mm, w:75mm
DATASET_V8_AXIAL_CROP_OFFSET_MM = [0, 0]  # h:0mm, w:0mm

# DATASET (v9)
DATASET_V9_SAGITTAL_CROP_MM = [35, 50]  # h:35mm, w:50mm
DATASET_V9_SAGITTAL_CROP_OFFSET_MM = [-5, 0]  # h:-5mm, w:0mm
DATASET_V9_AXIAL_CROP_MM = [45, 75]  # h:45mm, w:75mm
DATASET_V9_AXIAL_CROP_OFFSET_MM = [5, 0]  # h:5mm, w:0mm

# DATASET (v11)
DATASET_V11_SAGITTAL_CROP_MM = [[21.5, 14.5], [22.5, 22.5]]  # [top=21.5mm, bottom=14.5mm], [left=22.5mm, right=22.5mm]
DATASET_V11_SAGITTAL_CROP_OFFSET_MM = -5  # 法線ベクトル方向にmm単位でオフセットする
DATASET_V11_AXIAL_CROP_MM = [[17.5, 27.5], [37.5, 37.5]]  # [top=17.5mm, bottom=27.5mm], [left=37.5mm, right=37.5mm]

TTA_MODE = 'axial_flip'

# TEMPERATURE
SPINAL_T = 0.9092925874420968
LR_T = 0.9964720472122109

# MODEL INFO
ckpt_dir = SETTINGS.pretrained_checkpoint_dir if args.use_pretrain else SETTINGS.model_checkpoint_dir
print(f'Checkpoint directory: {ckpt_dir}')
SAGITTAL_KEYPOINT_MODEL_INFO = [
    {
        'config': 'src/stage1/moyashii/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v6/0002/exp0002_fold0_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/moyashii/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v6/0002/exp0002_fold1_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/moyashii/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v6/0003/exp0003_fold2_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0002/exp0002_fold0_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0002/exp0002_fold1_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0002/exp0002_fold2_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0002/exp0002_fold3_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0002_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0002/exp0002_fold4_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0003/exp0003_fold0_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0003/exp0003_fold1_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0003/exp0003_fold2_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0003/exp0003_fold3_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/sagittal/config/v6_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v6/0003/exp0003_fold4_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
]
AXIAL_KEYPOINT_MODEL_INFO = [
    {
        'config': 'src/stage1/moyashii/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v5/0003/exp0003_fold0_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/moyashii/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v5/0003/exp0003_fold1_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/moyashii/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/moyashii/v5/0003/exp0003_fold2_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v5/0003/exp0003_fold0_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v5/0003/exp0003_fold1_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v5/0003/exp0003_fold2_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v5/0003/exp0003_fold3_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
    {
        'config': 'src/stage1/tkmn/axial/config/v5_0003_config.yaml',
        'checkpoint': ckpt_dir / 'stage1/tkmn/v5/0003/exp0003_fold4_best_score.pth',
        'model_class': 'RSNA2024KeypointNetTest',
    },
]
RSNA2024_DATASET_VER2_MODEL_INFO = [
    # moyashii_exp0152_(effv2s)_224
    {
        'config': 'src/stage2/moyashii/config/0152.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0152/exp0152_fold1_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV8',
    },

    # moyashii_exp0176_v9(cntiny)_224
    {
        'config': 'src/stage2/moyashii/config/0176.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0176/exp0176_fold3_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0176.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0176/exp0176_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0188_v9(cntiny)_224
    {
        'config': 'src/stage2/moyashii/config/0188.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0188/exp0188_fold0_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0188.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0188/exp0188_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0196_v9(effb4)_224
    {
        'config': 'src/stage2/moyashii/config/0196.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0196/exp0196_fold2_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0196.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0196/exp0196_fold3_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0211_脊柱管強化モデルv9(cnnano)_320
    {
        'config': 'src/stage2/moyashii/config/0211.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0211/exp0211_fold0_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0211.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0211/exp0211_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # tkmn_spinal_cfg0123_aug04_convnext_tiny_cv012340.3994
    {
        'config': 'src/stage2/tkmn/config/S0123.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0123/expS0123_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0123.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0123/expS0123_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0141_aug02_efficientnetv2_rw_t_cv012340.3980
    {
        'config': 'src/stage2/tkmn/config/S0141.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0141/expS0141_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0141.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0141/expS0141_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0143_aug04_efficientnetv2_rw_t_cv012340.3998
    {
        'config': 'src/stage2/tkmn/config/S0143.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0143/expS0143_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0162_cw1_1_convnext_nano_cv012340.3968
    {
        'config': 'src/stage2/tkmn/config/S0162.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0162/expS0162_fold3_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0188_bs2_convnext_tiny_cv012340.3989
    {
        'config': 'src/stage2/tkmn/config/S0188.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0188/expS0188_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0188.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0188/expS0188_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0189_bs3_convnext_tiny_cv012340.3970
    {
        'config': 'src/stage2/tkmn/config/S0189.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0189/expS0189_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0189.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0189/expS0189_fold4_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0202_bs3_mnasnet_small_cv012340.4033
    {
        'config': 'src/stage2/tkmn/config/S0202.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0202/expS0202_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0204_bs2_resnet18_cv012340.4014
    {
        'config': 'src/stage2/tkmn/config/S0204.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0204/expS0204_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0204.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0204/expS0204_fold3_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_spinal_cfg0211_bs3_efficientnetv2_rw_t_cv012340.3979
    {
        'config': 'src/stage2/tkmn/config/S0211.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0211/expS0211_fold3_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/S0211.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/center/v9/0211/expS0211_fold4_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
]
RSNA2024_DATASET_VER6_MODEL_INFO = [
    # moyashii_0113 (ConvNext-Tiny (HCE), 224, 11 (pseudo))
    {
        'config': 'src/stage2/moyashii/config/0113.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0113/exp0113_fold0_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/moyashii/config/0113.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/center/v11/0113/exp0113_fold1_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNet',
    },
]
RSNA2024_DATASET_VER5_MODEL_INFO = [
    # moyashii_exp0082_LR(cnnano)_224
    {
        'config': 'src/stage2/moyashii/config/0082.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0082/exp0082_fold1_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNet',
    },

    # moyashii_exp0157_LRv9(effv2s)_224
    {
        'config': 'src/stage2/moyashii/config/0157.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0157/exp0157_fold1_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0157.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0157/exp0157_fold2_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0159_LRv9(cntiny)_224
    {
        'config': 'src/stage2/moyashii/config/0159.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0159/exp0159_fold0_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0177_LRv9(cntiny)_224
    {
        'config': 'src/stage2/moyashii/config/0177.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0177/exp0177_fold2_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0177.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0177/exp0177_fold3_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0191_LRv9(cntiny)_224
    {
        'config': 'src/stage2/moyashii/config/0191.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0191/exp0191_fold3_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0191.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0191/exp0191_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },


    # moyashii_exp0200_LRv9(cnnano)_224
    {
        'config': 'src/stage2/moyashii/config/0200.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0200/exp0200_fold0_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0200.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0200/exp0200_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # moyashii_exp0212_LRモデルv9(cnnano)_320
    {
        'config': 'src/stage2/moyashii/config/0212.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0212/exp0212_fold3_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },
    {
        'config': 'src/stage2/moyashii/config/0212.yaml',
        'checkpoint': ckpt_dir / 'stage2/moyashii/side/v11/0212/exp0212_fold4_best_score.pth',
        'dataset': 11,
        'model_class': 'RSNA2024AttentionNetV9',
    },

    # suguuuuu_exp2064_base-conv-batch2_fold_0.40268_submission
    {
        'config': 'src/stage2/suguuuuu/config/2064.yaml',
        'checkpoint': ckpt_dir / 'stage2/suguuuuu/side/v8/2064/exp2064_base-conv-batch2_fold2_best_score.pth',
        'dataset': 8,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/suguuuuu/config/2064.yaml',
        'checkpoint': ckpt_dir / 'stage2/suguuuuu/side/v8/2064/exp2064_base-conv-batch2_fold3_best_score.pth',
        'dataset': 8,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0101_aug00_convnext_nano_cv012340.4044
    {
        'config': 'src/stage2/tkmn/config/0101.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0101/exp0101_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/0101.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0101/exp0101_fold4_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0123_aug02_r160_resnet18_cv012340.4045
    {
        'config': 'src/stage2/tkmn/config/0123.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0123/exp0123_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/0123.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0123/exp0123_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0124_aug03_r160_resnet18_cv012340.4040
    {
        'config': 'src/stage2/tkmn/config/0124.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0124/exp0124_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0138_aug02_r256_maxvit_rmlp_nano_rw_256_cv012340.4032
    {
        'config': 'src/stage2/tkmn/config/0138.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0138/exp0138_fold0_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/0138.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0138/exp0138_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0139_aug03_r256_maxvit_rmlp_nano_rw_256_cv012340.4018
    {
        'config': 'src/stage2/tkmn/config/0139.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0139/exp0139_fold3_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/0139.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0139/exp0139_fold4_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },


    # tkmn_cfg0145_aug04_r256_maxvit_nano_rw_256_cv012340.4047
    {
        'config': 'src/stage2/tkmn/config/0145.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0145/exp0145_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
    {
        'config': 'src/stage2/tkmn/config/0145.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0145/exp0145_fold2_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },

    # tkmn_cfg0203_bs3_mnasnet_small_cv012340.4062
    {
        'config': 'src/stage2/tkmn/config/0203.yaml',
        'checkpoint': ckpt_dir / 'stage2/tkmn/side/v9/0203/exp0203_fold1_best_score.pth',
        'dataset': 9,
        'model_class': 'RSNA2024AttentionNet',
    },
]

# SUBMIT
CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
LEVELS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
PLANES = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']

# %% [markdown]
# ## Helper fucntions

# %%


def get_config(config_path: str, dot_list: list) -> dict:
    config_omega_from_yaml = OmegaConf.load(config_path)
    config_omega_from_args = OmegaConf.from_dotlist(dot_list)
    config_omega = OmegaConf.merge(config_omega_from_yaml, config_omega_from_args)
    config = OmegaConf.to_container(config_omega, resolve=True)  # DictConfig -> dict
    return config

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

# %% [markdown]
# ## Define models

# %%


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False, flatten=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps
        self.flatten = flatten

    def _gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        ret = self._gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            return ret[:, :, 0, 0]
        else:
            return ret

    def __repr__(self):
        if not isinstance(self.p, int):
            return (self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")
        else:
            return (self.__class__.__name__ + f"(p={self.p:.4f},eps={self.eps})")

# %%


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

        # 完全に無駄なModuleが含まれているのでコメントアウトしたけどGPU使用量減らなかった
        self.backbone = nn.Sequential(*layers)
        if "efficientnet" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif "swin" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif timm_model['model_name'].startswith('convnextv2_'):
            backbone_out_channels = base_model.num_features
        elif timm_model['model_name'].startswith('convnext_'):
            backbone_out_channels = base_model.head.in_features
        elif timm_model['model_name'].startswith('nfnet_'):
            backbone_out_channels = base_model.head.fc.in_features
        elif timm_model['model_name'].startswith('resnest'):
            backbone_out_channels = base_model.fc.in_features
        elif hasattr(timm_model['model_name'], "fc"):
            backbone_out_channels = base_model.fc.in_features
        else:
            backbone_out_channels = base_model.num_features
#             raise ValueError(f'{timm_model["model_name"]} is not supported.')

        # Attentionの設定
        self.transformer = RSNA2024TransformerBlock(input_dim=backbone_out_channels, num_heads=8, ff_hidden_dim=512)

        # 出力層
        self.fc = nn.Linear(backbone_out_channels, num_degenerative_conditions * num_segments * num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self,
                sagittal_t1_images: torch.Tensor,
                sagittal_t2_images: torch.Tensor,
                axial_t2_images: torch.Tensor,
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

        logits = self.fc(combined_features)
        logits = logits.view(-1, self._num_degenerative_conditions, self._num_segments, self._num_classes)
        outputs['logits'] = logits

        return outputs

# %%


class SliceAttentionBlock(nn.Module):
    # SENetのSEBlockを参考にしたスライスの重要度計算ブロック
    def __init__(self, feature_dim, reduction=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim // reduction, bias=False)
        self.fc2 = nn.Linear(feature_dim // reduction, 1, bias=False)  # 1にすることでスライスごとの重みを出力
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        excitation = self.fc1(x)  # (batch_size, num_slices, feature_dim) -> (batch_size, num_slices, feature_dim // reduction)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # (batch_size, num_slices, feature_dim // reduction) -> (batch_size, num_slices, 1)
        attention_weights = self.sigmoid(excitation)  # (batch_size, num_slices, 1)

        x = attention_weights * x  # (batch_size, num_slices, feature_dim)
        return x


class RSNA2024AttentionNetV8(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 num_classes: int = 3,
                 use_planes: list = ['sagittal_t1', 'sagittal_t2', 'axial_t2'],
                 ):
        super().__init__()
        self._num_classes = num_classes
        self._conditions = CONDITIONS
        self._use_planes = use_planes

        self._in_channels = timm_model.get('in_chans', 3)

        base_model = None
        self.backbones = nn.ModuleDict()
        self.gaps = nn.ModuleDict()
        for plane in self._use_planes:
            base_model = timm.create_model(**timm_model)
            layers = list(base_model.children())[:-2]
            self.backbones[plane] = nn.Sequential(*layers)
            self.gaps[plane] = GeM(flatten=True, p_trainable=True)

        if "efficientnet" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif "swin" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif timm_model['model_name'].startswith('convnext_'):
            backbone_out_channels = base_model.head.in_features
        elif timm_model['model_name'].startswith('nfnet_'):
            backbone_out_channels = base_model.head.fc.in_features
        elif timm_model['model_name'].startswith('resnest'):
            backbone_out_channels = base_model.fc.in_features
        elif hasattr(timm_model['model_name'], "fc"):
            backbone_out_channels = base_model.fc.in_features
        else:
            raise ValueError(f'{timm_model["model_name"]} is not supported.')

        # Attentionの設定
        self.transformer = RSNA2024TransformerBlock(input_dim=backbone_out_channels, num_heads=8, ff_hidden_dim=512)

        # Attention Weight and Head
        self.slice_attentions = nn.ModuleDict()
        self.fcs = nn.ModuleDict()
        for condition in self._conditions:
            self.slice_attentions[condition] = SliceAttentionBlock(backbone_out_channels)
            self.fcs[condition] = nn.Linear(backbone_out_channels, num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self,
                sagittal_t1_images: torch.Tensor,
                sagittal_t2_images: torch.Tensor,
                axial_t2_images: torch.Tensor,
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
            plane_backbone = self.backbones[plane]
            plane_gap = self.gaps[plane]

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
        combined_features = combined_features.permute(1, 0, 2)  # (batch_size, num_slices, feature_dim)

        # Apply Attention to weight slices based on importance
        logits_list = []
        for condition in self._conditions:
            attention_features = self.slice_attentions[condition](combined_features)  # (batch_size, num_slices, feature_dim)
            condition_features = attention_features.mean(dim=1)  # (batch_size, feature_dim)
            logits = self.fcs[condition](condition_features)
            logits_list.append(logits)
        logits = torch.stack(logits_list, dim=1)[:, :, None, :]  # (batch_size, num_degenerative_conditions, 1, num_classes)
        outputs['logits'] = logits

        return outputs


class RSNA2024AttentionNetV9(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 conditions: list[str] = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing',
                                          'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis'],
                 num_classes: int = 3,
                 use_planes: list[str] = ['sagittal_t1', 'sagittal_t2', 'axial_t2'],
                 ):
        super().__init__()
        self._num_classes = num_classes
        self._conditions = conditions
        self._use_planes = use_planes

        self._in_channels = timm_model.get('in_chans', 3)

        base_model = None
        self.backbones = nn.ModuleDict()
        self.gaps = nn.ModuleDict()
        for plane in self._use_planes:
            base_model = timm.create_model(**timm_model)
            layers = list(base_model.children())[:-2]
            self.backbones[plane] = nn.Sequential(*layers)
            self.gaps[plane] = GeM(flatten=True, p_trainable=True)

        self._model_name = timm_model['model_name']
        if "efficientnet" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif "swin" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        elif timm_model['model_name'].startswith('convnext_'):
            backbone_out_channels = base_model.head.in_features
        elif timm_model['model_name'].startswith('nfnet_'):
            backbone_out_channels = base_model.head.fc.in_features
        elif timm_model['model_name'].startswith('resnest'):
            backbone_out_channels = base_model.fc.in_features
        elif hasattr(timm_model['model_name'], "fc"):
            backbone_out_channels = base_model.fc.in_features
        else:
            raise ValueError(f'{timm_model["model_name"]} is not supported.')

        # Attentionの設定
        self.transformer = RSNA2024TransformerBlock(input_dim=backbone_out_channels, num_heads=8, ff_hidden_dim=512)

        # Attention Weight and Head
        self.slice_attentions = nn.ModuleDict()
        self.fcs = nn.ModuleDict()
        self.slice_fcs = nn.ModuleDict()
        for condition in self._conditions:
            self.slice_attentions[condition] = SliceAttentionBlock(backbone_out_channels)
            self.fcs[condition] = nn.Linear(backbone_out_channels, num_classes)
            self.slice_fcs[condition] = nn.Linear(backbone_out_channels, num_classes)

    def forward(self,
                sagittal_t1_images: torch.Tensor,
                sagittal_t2_images: torch.Tensor,
                axial_t2_images: torch.Tensor,
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
            plane_backbone = self.backbones[plane]
            plane_gap = self.gaps[plane]

            batch_size, plane_num_slices, h, w = plane_images.shape
            plane_images = plane_images.view(-1, self._in_channels, h, w)  # (batch_size * plane_num_slices, 1, h, w)
            plane_feats = plane_backbone(plane_images)  # (batch_size * plane_num_slices, feature_dim, h / 32, w / 32)
            if 'swinv2' in self._model_name:
                plane_feats = plane_feats.permute(0, 3, 1, 2)
            plane_feats = plane_gap(plane_feats)  # (batch_size * plane_num_slices, feature_dim)
            plane_feats = plane_feats.view(batch_size, plane_num_slices, -1)  # (batch_size, plane_num_slices, feature_dim)

            feats[plane] = plane_feats

        # 特徴量を結合して1つのシーケンスに統合
        combined_features = torch.cat(list(feats.values()), dim=1)  # (batch_size, num_slices, feature_dim)

        # Transformerを適用
        combined_features = combined_features.permute(1, 0, 2)  # (num_slices, batch_size, feature_dim)
        combined_features = self.transformer(combined_features)  # (num_slices, batch_size, feature_dim)
        combined_features = combined_features.permute(1, 0, 2)  # (batch_size, num_slices, feature_dim)

        # Apply Attention to weight slices based on importance
        logits_list = []
        for condition in self._conditions:
            attention_features = self.slice_attentions[condition](combined_features)  # (batch_size, num_slices, feature_dim)
            condition_features = attention_features.mean(dim=1)  # (batch_size, feature_dim)
            logits = self.fcs[condition](condition_features)  # (batch_size, num_classes)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=1)[:, :, None, :]  # (batch_size, num_degenerative_conditions, 1, num_classes)
        outputs['logits'] = logits

        return outputs

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

# %% [markdown]
# ## Detectors

# %%


def predict_with_axial_flip(model, sagittal_t1_image, sagittal_t2_image, axial_t2_image):
    # 元のaxial画像で予測
    with torch.no_grad():
        original_logits = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']

    # axial画像を左右反転
    flipped_axial_t2_image = torch.flip(axial_t2_image, dims=[3])

    # 左右反転したaxial画像で予測
    with torch.no_grad():
        flipped_logits = model(sagittal_t1_image, sagittal_t2_image, flipped_axial_t2_image)['logits']

    # 元のaxial画像と反転したaxial画像の予測結果を平均
    averaged_logits = (original_logits + flipped_logits) / 2.0

    return averaged_logits


class RSNA2024DatasetVer2Detector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 temperature: float,
                 levels: list = LEVELS,
                 condition: list = CONDITIONS):
        self._model_infos = model_infos
        self._device = device
        self._temperature = temperature
        self._levels = levels
        self._condition = condition
        self._prediction_infos = []

        for model_info in model_infos:
            # model
            config = get_config(model_info['config'], [])
            if 'name' in config['model']:
                config['model'].pop('name')
            config['model']['timm_model']['pretrained'] = False
            model_class = globals()[model_info['model_class']]
            model = model_class(**config['model'])
            checkpoint = model_info['checkpoint']
            state_dict = torch.load(checkpoint, map_location=device)
            for k in ['target_loss.weight', 'target_loss.ce_loss.weight', 'target_loss.ce_loss.class_weights', 'target_loss.ce_loss.distance_matrix',
                      "level_fc.weight", "level_fc.bias", "target_loss.condition_weight", "target_loss.slice_ce_loss.weight"]:
                if k in state_dict:
                    del state_dict[k]
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            # dataset
            dataset_version = model_info['dataset']

            # image shape
            image_shapes = config['dataset']['image_shapes']

            # transform
            transforms = {}
            for plane in config['transform']:
                transforms[plane] = A.Compose([
                    A.Resize(*config['transform'][plane]['image_size']),
                    A.Normalize(mean=0.5, std=0.5)
                ])

            self._prediction_infos.append(dict(
                model=model,
                dataset_version=dataset_version,
                input_shapes=image_shapes,
                transforms=transforms,
            ))

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

    def _preprocess(self,
                    study_image_df: pd.DataFrame,
                    input_shapes: dict,
                    transforms: dict,
                    device: torch.device) -> pd.DataFrame:
        plane_images = {}
        for plane in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
            image_shape = input_shapes[plane]
            transform = transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane]

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            if plane == 'Axial T2':
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            else:
                plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)
            images = self._select_n_elements(plane_study_image_df['image'].values, plane_channel)
            if len(images) > 0:
                for i, image in enumerate(images):
                    plane_image[..., i] = cv2.resize(image, image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        sagittal_t1_image = torch.Tensor(plane_images['Sagittal T1'][None, ...]).to(device)
        sagittal_t2_image = torch.Tensor(plane_images['Sagittal T2/STIR'][None, ...]).to(device)
        axial_t2_image = torch.Tensor(plane_images['Axial T2'][None, ...]).to(device)
        return sagittal_t1_image, sagittal_t2_image, axial_t2_image

    def predict(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        return self._predict_multi(study_id, study_image_df)

    def _predict_single(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds_per_model = []
        row_names_per_model = []
        for i, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shape = prediction_info['input_shapes']
            transform = prediction_info['transforms']

            preds = []
            row_names = []
            for level in self._levels:
                _level = level.lower().replace('/', '_')
                sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                    study_image_df[(study_image_df['level'] == level) & (study_image_df['dataset_version'] == dataset_version)],
                    input_shape,
                    transform,
                    self._device)
                with torch.no_grad():
                    logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
                    pred = logit.softmax(dim=-1).cpu().numpy().reshape(len(self._condition), 3)
                    preds.append(pred)
                    for condition in self._condition:
                        row_names.append(str(study_id) + '_' + condition + '_' + _level)
            preds_per_model.append(np.concatenate(preds, axis=0))
            row_names_per_model.append(row_names)

        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names_per_model[0]
        pred_df[['normal_mild', 'moderate', 'severe']] = np.mean(preds_per_model, axis=0)
        return pred_df

    def _predict_multi(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        row_names = []

        for _, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shape = prediction_info['input_shapes']
            transform = prediction_info['transforms']
            dataset_study_image_df = study_image_df[(study_image_df['dataset_version'] == dataset_version)]

            sagittal_t1_image_list = []
            sagittal_t2_image_list = []
            axial_t2_image_list = []
            for level in self._levels:
                sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                    dataset_study_image_df[(dataset_study_image_df['level'] == level)],
                    input_shape,
                    transform,
                    'cpu')
                sagittal_t1_image_list.append(sagittal_t1_image)
                sagittal_t2_image_list.append(sagittal_t2_image)
                axial_t2_image_list.append(axial_t2_image)

            sagittal_t1_image = torch.cat(sagittal_t1_image_list, dim=0).to(self._device)
            sagittal_t2_image = torch.cat(sagittal_t2_image_list, dim=0).to(self._device)
            axial_t2_image = torch.cat(axial_t2_image_list, dim=0).to(self._device)

            if TTA_MODE == 'axial_flip':
                logit = predict_with_axial_flip(model, sagittal_t1_image, sagittal_t2_image, axial_t2_image)
            else:
                with torch.no_grad():
                    logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
            pred = (logit / self._temperature).softmax(dim=-1).cpu().numpy()
            preds.append(pred)

            for level in self._levels:
                _level = level.lower().replace('/', '_')
                for condition in self._condition:
                    row_names.append(str(study_id) + '_' + condition + '_' + _level)

        preds = np.concatenate(preds, axis=0).reshape(-1, 3)
        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names
        pred_df[['normal_mild', 'moderate', 'severe']] = preds
        pred_df = pred_df.groupby('row_id').mean().reset_index()
        return pred_df

# %%


class RSNA2024DatasetVer5Detector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 temperature: float,
                 levels: list = LEVELS,
                 condition: list = ['neural_foraminal_narrowing', 'subarticular_stenosis']):
        self._model_infos = model_infos
        self._device = device
        self._temperature = temperature
        self._levels = levels
        self._condition = condition
        self._prediction_infos = []

        for model_info in model_infos:
            # model
            config = get_config(model_info['config'], [])
            if 'name' in config['model']:
                config['model'].pop('name')
            config['model']['timm_model']['pretrained'] = False
            model_class = globals()[model_info['model_class']]
            model = model_class(**config['model'])
            checkpoint = model_info['checkpoint']
            state_dict = torch.load(checkpoint, map_location=device)
            for k in ['target_loss.weight', 'target_loss.ce_loss.weight', 'target_loss.ce_loss.class_weights', 'target_loss.ce_loss.distance_matrix',
                      "level_fc.weight", "level_fc.bias", "target_loss.condition_weight", "target_loss.slice_ce_loss.weight"]:
                if k in state_dict:
                    del state_dict[k]
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            # dataset
            dataset_version = model_info['dataset']

            # image shape
            image_shapes = config['dataset']['image_shapes']

            # transform
            transforms = {}
            for plane in config['transform']:
                transforms[plane] = A.Compose([
                    A.Resize(*config['transform'][plane]['image_size']),
                    A.Normalize(mean=0.5, std=0.5)
                ])

            self._prediction_infos.append(dict(
                model=model,
                dataset_version=dataset_version,
                input_shapes=image_shapes,
                transforms=transforms,
            ))

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

    def _preprocess(self,
                    study_image_df: pd.DataFrame,
                    side: str,
                    input_shapes: dict,
                    transforms: dict,
                    device: torch.device) -> tuple:
        plane_images = {}
        for plane in ['Sagittal T1', 'Sagittal T2/STIR']:
            image_shape = input_shapes[plane]
            transform = transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane]

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
            images = self._select_n_elements(plane_study_image_df['image'].values, plane_channel)
            if len(images) > 0:
                for i, image in enumerate(images):
                    plane_image[..., i] = cv2.resize(image, image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        for plane in ['Axial T2']:
            image_shape = input_shapes[plane]
            transform = transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane]

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            images = self._select_n_elements(plane_study_image_df['image'].values, plane_channel)

            if len(images) > 0:
                for i, image in enumerate(images):
                    h, w = image.shape
                    if side == 'right':
                        crop_w = (0.0, 0.6)
                        crop_image = image[:, int(w * crop_w[0]):int(w * crop_w[1])]
                        crop_image = cv2.flip(crop_image, 1)  # 向きをそろえる。
                    else:
                        crop_w = (0.4, 1.0)
                        crop_image = image[:, int(w * crop_w[0]):int(w * crop_w[1])]
                    plane_image[..., i] = cv2.resize(crop_image, image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        sagittal_t1_image = torch.Tensor(plane_images['Sagittal T1'][None, ...]).to(device)
        sagittal_t2_image = torch.Tensor(plane_images['Sagittal T2/STIR'][None, ...]).to(device)
        axial_t2_image = torch.Tensor(plane_images['Axial T2'][None, ...]).to(device)
        return sagittal_t1_image, sagittal_t2_image, axial_t2_image

    def predict(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        return self._predict_multi(study_id, study_image_df)

    def _predict_single(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds_per_model = []
        row_names_per_model = []
        for i, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shapes = prediction_info['input_shapes']
            transform = prediction_info['transforms']

            preds = []
            row_names = []
            for side in ['left', 'right']:
                for level in self._levels:
                    _level = level.lower().replace('/', '_')
                    sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                        study_image_df[(study_image_df['level'] == level) & (study_image_df['dataset_version'] == dataset_version)],
                        side,
                        input_shapes,
                        transform,
                        self._device)
                    with torch.no_grad():
                        logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
                        pred = logit.softmax(dim=-1).cpu().numpy().reshape(len(self._condition), 3)
                        preds.append(pred)
                        for condition in self._condition:
                            row_names.append(str(study_id) + '_' + side + '_' + condition + '_' + _level)
            preds_per_model.append(np.concatenate(preds, axis=0))
            row_names_per_model.append(row_names)

        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names_per_model[0]
        pred_df[['normal_mild', 'moderate', 'severe']] = np.mean(preds_per_model, axis=0)
        return pred_df

    def _predict_multi(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        row_names = []

        for _, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shapes = prediction_info['input_shapes']
            transform = prediction_info['transforms']
            dataset_study_image_df = study_image_df[(study_image_df['dataset_version'] == dataset_version)]

            for side in ['left', 'right']:
                sagittal_t1_image_list = []
                sagittal_t2_image_list = []
                axial_t2_image_list = []
                for level in self._levels:
                    sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                        dataset_study_image_df[(dataset_study_image_df['level'] == level)],
                        side,
                        input_shapes,
                        transform,
                        'cpu')
                    sagittal_t1_image_list.append(sagittal_t1_image)
                    sagittal_t2_image_list.append(sagittal_t2_image)
                    axial_t2_image_list.append(axial_t2_image)

                sagittal_t1_image = torch.cat(sagittal_t1_image_list, dim=0).to(self._device)
                sagittal_t2_image = torch.cat(sagittal_t2_image_list, dim=0).to(self._device)
                axial_t2_image = torch.cat(axial_t2_image_list, dim=0).to(self._device)

                with torch.no_grad():
                    logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
                    pred = (logit / self._temperature).softmax(dim=-1).cpu().numpy()
                    preds.append(pred)

                for level in self._levels:
                    _level = level.lower().replace('/', '_')
                    for condition in self._condition:
                        row_names.append(str(study_id) + '_' + side + '_' + condition + '_' + _level)

        preds = np.concatenate(preds, axis=0).reshape(-1, 3)
        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names
        pred_df[['normal_mild', 'moderate', 'severe']] = preds
        pred_df = pred_df.groupby('row_id').mean().reset_index()
        return pred_df

# %%


class RSNA2024DatasetVer6Detector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 temperature: float,
                 levels: list = LEVELS,
                 condition: list = ['spinal_canal_stenosis']):
        self._model_infos = model_infos
        self._device = device
        self._temperature = temperature
        self._levels = levels
        self._condition = condition
        self._prediction_infos = []

        for model_info in model_infos:
            # model
            config = get_config(model_info['config'], [])
            if 'name' in config['model']:
                config['model'].pop('name')
            config['model']['timm_model']['pretrained'] = False
            model_class = globals()[model_info['model_class']]
            model = model_class(**config['model'])
            checkpoint = model_info['checkpoint']
            state_dict = torch.load(checkpoint, map_location=device)
            for k in ['target_loss.weight', 'target_loss.ce_loss.weight', 'target_loss.ce_loss.class_weights', 'target_loss.ce_loss.distance_matrix',
                      "level_fc.weight", "level_fc.bias", "target_loss.condition_weight", "target_loss.slice_ce_loss.weight"]:
                if k in state_dict:
                    del state_dict[k]
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            # dataset
            dataset_version = model_info['dataset']

            # image shape
            image_shapes = config['dataset']['image_shapes']

            # transform
            transforms = {}
            for plane in config['transform']:
                transforms[plane] = A.Compose([
                    A.Resize(*config['transform'][plane]['image_size']),
                    A.Normalize(mean=0.5, std=0.5)
                ])

            # sagittal start end pos
            sagittal_start_pos = config['dataset'].get('sagittal_start_pos', 0.25)
            sagittal_end_pos = config['dataset'].get('sagittal_end_pos', 0.75)

            self._prediction_infos.append(dict(
                model=model,
                dataset_version=dataset_version,
                input_shapes=image_shapes,
                transforms=transforms,
                sagittal_start_pos=sagittal_start_pos,
                sagittal_end_pos=sagittal_end_pos,
            ))

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

    def _preprocess(self,
                    study_image_df: pd.DataFrame,
                    input_shapes: dict,
                    transforms: dict,
                    sagittal_start_pos: float,
                    sagittal_end_pos: float,
                    device: torch.device) -> pd.DataFrame:

        plane_images = {}
        for plane in ['Sagittal T1', 'Sagittal T2/STIR']:
            image_shape = input_shapes[plane]
            transform = transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=True)

            # 体の中心近傍のスライスを取得する
            start_idx = math.floor(len(plane_study_image_df) * sagittal_start_pos)
            end_idx = math.ceil(len(plane_study_image_df) * sagittal_end_pos)
            plane_study_image_df = plane_study_image_df.iloc[start_idx:end_idx]

            images = self._select_n_elements(plane_study_image_df['image'].values, plane_channel)
            if len(images) > 0:
                for i, image in enumerate(images):
                    plane_image[..., i] = cv2.resize(image, image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        for plane in ['Axial T2']:
            image_shape = input_shapes[plane]
            transform = transforms[plane]
            plane_image = np.zeros(image_shape, dtype=np.uint8)
            plane_channel = image_shape[-1]
            plane_study_image_df = study_image_df[study_image_df['series_description'] == plane].copy()

            # 複数のシリーズが存在する場合はランダムに選択
            if len(plane_study_image_df['series_id'].unique()) > 1:
                selected_series_id = np.random.choice(plane_study_image_df['series_id'].unique())
                plane_study_image_df = plane_study_image_df[plane_study_image_df['series_id'] == selected_series_id]

            plane_study_image_df = plane_study_image_df.sort_values('standard_pos', ascending=False)
            images = self._select_n_elements(plane_study_image_df['image'].values, plane_channel)
            if len(images) > 0:
                for i, image in enumerate(images):
                    plane_image[..., i] = cv2.resize(image, image_shape[:2])
                if transform is not None:
                    plane_image = transform(image=plane_image)['image']
            plane_images[plane] = plane_image.transpose(2, 0, 1).astype(np.float32)

        sagittal_t1_image = torch.Tensor(plane_images['Sagittal T1'][None, ...]).to(device)
        sagittal_t2_image = torch.Tensor(plane_images['Sagittal T2/STIR'][None, ...]).to(device)
        axial_t2_image = torch.Tensor(plane_images['Axial T2'][None, ...]).to(device)
        return sagittal_t1_image, sagittal_t2_image, axial_t2_image

    def predict(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        return self._predict_multi(study_id, study_image_df)

    def _predict_single(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds_per_model = []
        row_names_per_model = []
        for i, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shapes = prediction_info['input_shapes']
            transform = prediction_info['transforms']
            sagittal_start_pot = prediction_info['sagittal_start_pos']
            sagittal_end_pos = prediction_info['sagittal_end_pos']

            preds = []
            row_names = []
            for level in self._levels:
                _level = level.lower().replace('/', '_')
                sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                    study_image_df[(study_image_df['level'] == level) & (study_image_df['dataset_version'] == dataset_version)],
                    input_shapes,
                    transform,
                    sagittal_start_pot,
                    sagittal_end_pos,
                    self._device)
                with torch.no_grad():
                    logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
                    pred = logit.softmax(dim=-1).cpu().numpy().reshape(len(self._condition), 3)
                    preds.append(pred)
                    for condition in self._condition:
                        row_names.append(str(study_id) + '_' + condition + '_' + _level)
            preds_per_model.append(np.concatenate(preds, axis=0))
            row_names_per_model.append(row_names)

        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names_per_model[0]
        pred_df[['normal_mild', 'moderate', 'severe']] = np.mean(preds_per_model, axis=0)
        return pred_df

    def _predict_multi(self, study_id: int, study_image_df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        row_names = []
        for i, prediction_info in enumerate(self._prediction_infos):
            model = prediction_info['model']
            dataset_version = prediction_info['dataset_version']
            input_shapes = prediction_info['input_shapes']
            transform = prediction_info['transforms']
            sagittal_start_pot = prediction_info['sagittal_start_pos']
            sagittal_end_pos = prediction_info['sagittal_end_pos']
            dataset_study_image_df = study_image_df[(study_image_df['dataset_version'] == dataset_version)]

            sagittal_t1_image_list = []
            sagittal_t2_image_list = []
            axial_t2_image_list = []
            for level in self._levels:
                _level = level.lower().replace('/', '_')
                sagittal_t1_image, sagittal_t2_image, axial_t2_image = self._preprocess(
                    dataset_study_image_df[(dataset_study_image_df['level'] == level)],
                    input_shapes,
                    transform,
                    sagittal_start_pot,
                    sagittal_end_pos,
                    'cpu')
                sagittal_t1_image_list.append(sagittal_t1_image)
                sagittal_t2_image_list.append(sagittal_t2_image)
                axial_t2_image_list.append(axial_t2_image)
            sagittal_t1_image = torch.cat(sagittal_t1_image_list, dim=0).to(self._device)
            sagittal_t2_image = torch.cat(sagittal_t2_image_list, dim=0).to(self._device)
            axial_t2_image = torch.cat(axial_t2_image_list, dim=0).to(self._device)
            if TTA_MODE == 'axial_flip':
                logit = predict_with_axial_flip(model, sagittal_t1_image, sagittal_t2_image, axial_t2_image)
            else:
                with torch.no_grad():
                    logit = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image)['logits']
            pred = (logit / self._temperature).softmax(dim=-1).cpu().numpy()
            preds.append(pred)

            for level in self._levels:
                _level = level.lower().replace('/', '_')
                for condition in self._condition:
                    row_names.append(str(study_id) + '_' + condition + '_' + _level)

        preds = np.concatenate(preds, axis=0).reshape(-1, 3)
        pred_df = pd.DataFrame()
        pred_df['row_id'] = row_names
        pred_df[['normal_mild', 'moderate', 'severe']] = preds
        pred_df = pred_df.groupby('row_id').mean().reset_index()
        return pred_df

# %%


class RSNA2024SagittalKeypointDetector:
    def __init__(self,
                 model_infos: list,
                 device: torch.device,
                 levels: list = LEVELS,
                 condition: list = CONDITIONS,
                 planes: list = PLANES):
        self._model_infos = model_infos
        self._device = device
        self._levels = levels
        self._condition = condition
        self._planes = planes
        self._prediction_infos = []

        for model_info in model_infos:
            config_path = model_info['config']
            checkpoint = model_info['checkpoint']

            config = get_config(config_path, [])
            config['model']['timm_model']['pretrained'] = False
            model_class = globals()[model_info['model_class']]
            model = model_class(**config['model'])
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            model.to(device)

            transforms = A.Compose([
                A.Resize(*config['transform']['image_size']),
                A.Normalize(mean=0.5, std=0.5)
            ])

            input_channels = config['dataset'].get('num_slices', 1)
            self._prediction_infos.append(dict(
                model=model,
                transforms=transforms,
                input_channels=input_channels,
            ))

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
            image = stack['arrays'][idx]
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
        for prediction_info in self._prediction_infos:
            model = prediction_info['model']
            transform = prediction_info['transforms']
            input_channel = prediction_info['input_channels']

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
                 levels: list = LEVELS,
                 condition: list = CONDITIONS,
                 planes: list = PLANES):
        self._model_infos = model_infos
        self._device = device
        self._levels = levels
        self._condition = condition
        self._planes = planes
        self._prediction_infos = []

        for model_info in model_infos:
            config_path = model_info['config']
            checkpoint = model_info['checkpoint']

            config = get_config(config_path, [])
            config['model']['timm_model']['pretrained'] = False
            model_class = globals()[model_info['model_class']]
            model = model_class(**config['model'])
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            model.to(device)

            transforms = A.Compose([
                A.Resize(*config['transform']['image_size']),
                A.Normalize(mean=0.5, std=0.5)
            ])

            input_channels = config['dataset'].get('num_slices', 1)

            self._prediction_infos.append(dict(
                model=model,
                transforms=transforms,
                input_channels=input_channels,
            ))

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
            image = stack['arrays'][idx]
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
        for prediction_info in self._prediction_infos:
            model = prediction_info['model']
            transform = prediction_info['transforms']
            input_channel = prediction_info['input_channels']

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

# %% [markdown]
# ## Read DICOM

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

    arrays = [apply_min_max_norm(array) for array in arrays]

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
# ## Grouping Axial

# %%
# Function to calculate the midpoint between two points


def midpoint(p1, p2):
    return tuple((np.array(p1) + np.array(p2)) / 2)


# Function to calculate an extended point
def extended_point(p1, p2, direction='outward'):
    vector = np.array(p2) - np.array(p1)
    extended = np.array(p1) - vector
    return tuple(extended)


def extend_sagment_point(points):
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

    # Check if the intersection point lies on the line segment
    return 0 <= t <= 1

# %% [markdown]
# ## Crop angle adjustment function using spline curves

# %%
# Distance-based spline interpolation


def spline_interpolation(x_coords, y_coords, num_points=100, spline_degree=2, smoothing_factor=1):
    # 座標に基づく累積距離を計算
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

    # 累積距離に対してスプライン補間を適用
    spline_x = UnivariateSpline(cumulative_distances, x_coords, k=spline_degree, s=smoothing_factor)
    spline_y = UnivariateSpline(cumulative_distances, y_coords, k=spline_degree, s=smoothing_factor)

    # 補間用の新しい距離ベースの点を生成
    interp_distances = np.linspace(0, cumulative_distances[-1], num_points)
    x_smooth = spline_x(interp_distances)
    y_smooth = spline_y(interp_distances)

    return x_smooth, y_smooth, interp_distances


def extract_and_correct_rectangle(image, rect_corners):
    # 長方形の左上、右上、右下の3点を使ってアフィン変換行列を計算
    width = int(np.linalg.norm(rect_corners[0] - rect_corners[1]))  # 長方形の幅
    height = int(np.linalg.norm(rect_corners[1] - rect_corners[2]))  # 長方形の高さ

    src_pts = rect_corners[:3].astype(np.float32)
    dst_pts = np.array([[0, 0], [width, 0], [width, height]], dtype=np.float32)

    # アフィン変換行列を計算
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # 画像をアフィン変換して回転補正
    warped = cv2.warpAffine(image, M, (width, height))

    return warped


def extract_rectangle_from_single_point(
    image,
    x_smooth,
    y_smooth,
    interp_distances,
    x_coord,
    y_coord,
    crop_top,
    crop_bottom,
    crop_left,
    crop_right,
    offset_mm,
    pixel_spacing,
    spline_degree=2,
    smoothing_factor=1
):
    # スプラインの各点における接線の傾きを計算
    spline_x_deriv = UnivariateSpline(interp_distances, x_smooth, k=spline_degree, s=smoothing_factor).derivative()
    spline_y_deriv = UnivariateSpline(interp_distances, y_smooth, k=spline_degree, s=smoothing_factor).derivative()

    # 赤い点に最も近いスプライン補完の点を探す
    distances = np.sqrt((x_smooth - x_coord)**2 + (y_smooth - y_coord)**2)
    i = np.argmin(distances)  # 最も近い点のインデックス

    # 接線の傾きを取得
    dx = spline_x_deriv(interp_distances[i])
    dy = spline_y_deriv(interp_distances[i])

    # 法線の傾きは接線の傾きの負の逆数
    normal_slope = -dx / dy

    # 法線の角度（ラジアン）を計算
    angle = np.arctan(normal_slope)

    # 長方形の中心を(x_coord, y_coord)として、回転させた四隅の座標を計算
    rect_corners = np.array([
        [-crop_left, -crop_top],
        [crop_right, -crop_top],
        [crop_right, crop_bottom],
        [-crop_left, crop_bottom]
    ])

    # 回転行列を使って長方形を回転
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_corners = rect_corners.dot(rotation_matrix.T)

    # 長方形の四隅の座標を計算し、中心座標(x_coord, y_coord)に移動
    rect_x = rotated_corners[:, 0] + x_coord
    rect_y = rotated_corners[:, 1] + y_coord

    # オフセット量を法線方向に適用
    offset_x_mm = offset_mm * np.cos(angle)
    offset_y_mm = offset_mm * np.sin(angle)
    rect_x += offset_x_mm / pixel_spacing[1]
    rect_y += offset_y_mm / pixel_spacing[0]

    # 長方形領域を切り出して回転補正
    rect_corners = np.column_stack((rect_x, rect_y))
    corrected_image = extract_and_correct_rectangle(image, rect_corners)

    return corrected_image, rect_x, rect_y

# %% [markdown]
# ## Create Detectors


# %%
print('Create keypoint detectors')
SAGITTAL_KEYPOINT_DETECTOR = RSNA2024SagittalKeypointDetector(model_infos=SAGITTAL_KEYPOINT_MODEL_INFO, device=DEVICE0)
AXIAL_KEYPOINT_DETECTOR = RSNA2024AxialKeypointDetector(model_infos=AXIAL_KEYPOINT_MODEL_INFO, device=DEVICE0)
print('Create classifiers')
LEVEL_SPLIT_DETECTOR = RSNA2024DatasetVer2Detector(model_infos=RSNA2024_DATASET_VER2_MODEL_INFO, device=DEVICE0, temperature=SPINAL_T)
LEVEL_SPINAL_SPLIT_DETECTOR = RSNA2024DatasetVer6Detector(model_infos=RSNA2024_DATASET_VER6_MODEL_INFO, device=DEVICE0, temperature=SPINAL_T)
LEVEL_LR_SPLIT_DETECTOR = RSNA2024DatasetVer5Detector(model_infos=RSNA2024_DATASET_VER5_MODEL_INFO, device=DEVICE0, temperature=LR_T)

# %% [markdown]
# ## Prediction

# %%


def create_dataset_v8_sagittal(
    stack: dict,
    study_id: int,
    series_id: int,
    desc: str,
    sagittal_image_norm_coord: dict,
    sagittal_crop_mm: list[float],
    sagittal_crop_offset_mm: list[float],
) -> list[dict]:
    data = []
    num_slices = len(stack['instance_numbers'])
    for i in range(num_slices):
        image = stack['arrays'][i]
        pixel_spacing = stack['pixel_spacings'][i]
        crop_y_half, crop_x_half = (sagittal_crop_mm / pixel_spacing / 2).tolist()
        crop_y_offset, crop_x_offset = (sagittal_crop_offset_mm / pixel_spacing).tolist()
        image_shape = stack['image_shapes'][i]
        for level, position in sagittal_image_norm_coord.items():
            x, y = position[0] * image_shape[1], position[1] * image_shape[0]
            left = round(x - crop_x_half + crop_x_offset)
            right = round(x + crop_x_half + crop_x_offset)
            top = round(y - crop_y_half + crop_y_offset)
            bottom = round(y + crop_y_half + crop_y_offset)
            if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
                print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
                continue
            crop_image = image[top:bottom, left:right]

            data.append(dict(
                study_id=study_id,
                series_id=series_id,
                series_description=desc,
                level=level,
                instance_no=stack['instance_numbers'][i],
                standard_pos=stack['standard_positions'][i],
                image=crop_image,
                dataset_version=8,
            ))

    return data


def create_dataset_v9_sagittal(
    stack: dict,
    study_id: int,
    series_id: int,
    desc: str,
    sagittal_image_norm_coord: dict,
    sagittal_crop_mm: list[float],
    sagittal_crop_offset_mm: list[float],
) -> list[dict]:
    data = []
    num_slices = len(stack['instance_numbers'])
    for i in range(num_slices):
        image = stack['arrays'][i]
        pixel_spacing = stack['pixel_spacings'][i]
        crop_y_half, crop_x_half = (sagittal_crop_mm / pixel_spacing / 2).tolist()
        crop_y_offset, crop_x_offset = (sagittal_crop_offset_mm / pixel_spacing).tolist()
        image_shape = stack['image_shapes'][i]
        for level, position in sagittal_image_norm_coord.items():
            x, y = position[0] * image_shape[1], position[1] * image_shape[0]
            left = round(x - crop_x_half + crop_x_offset)
            right = round(x + crop_x_half + crop_x_offset)
            top = round(y - crop_y_half + crop_y_offset)
            bottom = round(y + crop_y_half + crop_y_offset)
            if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
                print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
                continue
            crop_image = image[top:bottom, left:right]

            data.append(dict(
                study_id=study_id,
                series_id=series_id,
                series_description=desc,
                level=level,
                instance_no=stack['instance_numbers'][i],
                standard_pos=stack['standard_positions'][i],
                image=crop_image,
                dataset_version=9,
            ))

    return data


def create_dataset_v11_sagittal(
    stack: dict,
    study_id: int,
    series_id: int,
    desc: str,
    sagittal_image_norm_coord: dict,
    sagittal_crop_mm: list[float],
    sagittal_crop_offset_mm: list[float],
) -> list[dict]:
    data = []
    num_slices = len(stack['instance_numbers'])
    for i in range(num_slices):
        image = stack['arrays'][i]
        pixel_spacing = stack['pixel_spacings'][i]
        image_shape = stack['image_shapes'][i]

        x_coords = []
        y_coords = []
        for norm_x, norm_y in sagittal_image_norm_coord.values():
            x_coords.append(norm_x * image_shape[1])
            y_coords.append(norm_y * image_shape[0])
        x_smooth, y_smooth, interp_distances = spline_interpolation(x_coords, y_coords)

        (crop_top, crop_bottom), (crop_left, crop_right) = (np.array(sagittal_crop_mm) / pixel_spacing).tolist()
        for level, position in sagittal_image_norm_coord.items():
            x, y = position[0] * image_shape[1], position[1] * image_shape[0]
            crop_image, _, _ = extract_rectangle_from_single_point(image,
                                                                   x_smooth, y_smooth,
                                                                   interp_distances,
                                                                   x, y,
                                                                   crop_top, crop_bottom, crop_left, crop_right,
                                                                   sagittal_crop_offset_mm, pixel_spacing)

            data.append(dict(
                study_id=study_id,
                series_id=series_id,
                series_description=desc,
                level=level,
                instance_no=stack['instance_numbers'][i],
                standard_pos=stack['standard_positions'][i],
                image=crop_image,
                dataset_version=11,
            ))

    return data


def create_dataset_v8_axial(
    stack: dict,
    stack_indices: list[int],
    study_id: int,
    series_id: int,
    level: str,
    desc: str,
    axial_image_norm_coord: dict,
    axial_crop_mm: list[float],
    axial_crop_offset_mm: list[float],
) -> list[dict]:
    data = []
    for i in stack_indices:
        image = stack['arrays'][i]
        image_shape = stack['image_shapes'][i]
        pixel_spacing = stack['pixel_spacings'][i]
        crop_y_half, crop_x_half = (axial_crop_mm / pixel_spacing / 2).tolist()
        crop_y_offset, crop_x_offset = (axial_crop_offset_mm / pixel_spacing).tolist()

        x, y = axial_image_norm_coord['center'][0] * image_shape[1], axial_image_norm_coord['center'][1] * image_shape[0]
        left = round(x - crop_x_half + crop_x_offset)
        right = round(x + crop_x_half + crop_x_offset)
        top = round(y - crop_y_half + crop_y_offset)
        bottom = round(y + crop_y_half + crop_y_offset)
        if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
            print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
            continue

        crop_image = image[top:bottom, left:right]
        data.append(dict(
            study_id=study_id,
            series_id=series_id,
            series_description=desc,
            level=level,
            instance_no=stack['instance_numbers'][i],
            standard_pos=stack['standard_positions'][i],
            image=crop_image,
            dataset_version=8,
        ))

    return data


def create_dataset_v9_axial(
    stack: dict,
    stack_indices: list[int],
    study_id: int,
    series_id: int,
    level: str,
    desc: str,
    axial_image_norm_coord: dict,
    axial_crop_mm: list[float],
    axial_crop_offset_mm: list[float],
) -> list[dict]:
    data = []
    for i in stack_indices:
        image = stack['arrays'][i]
        image_shape = stack['image_shapes'][i]
        pixel_spacing = stack['pixel_spacings'][i]
        crop_y_half, crop_x_half = (axial_crop_mm / pixel_spacing / 2).tolist()
        crop_y_offset, crop_x_offset = (axial_crop_offset_mm / pixel_spacing).tolist()

        x, y = axial_image_norm_coord['center'][0] * image_shape[1], axial_image_norm_coord['center'][1] * image_shape[0]
        left = round(x - crop_x_half + crop_x_offset)
        right = round(x + crop_x_half + crop_x_offset)
        top = round(y - crop_y_half + crop_y_offset)
        bottom = round(y + crop_y_half + crop_y_offset)
        if left < 0 or right >= image_shape[1] or top < 0 or bottom >= image_shape[0]:
            print(f'{study_id} {series_id} {level} {stack["instance_numbers"][i]} is out of image')
            continue

        crop_image = image[top:bottom, left:right]
        data.append(dict(
            study_id=study_id,
            series_id=series_id,
            series_description=desc,
            level=level,
            instance_no=stack['instance_numbers'][i],
            standard_pos=stack['standard_positions'][i],
            image=crop_image,
            dataset_version=9,
        ))

    return data


def create_dataset_v11_axial(
    stack: dict,
    stack_indices: list[int],
    study_id: int,
    series_id: int,
    level: str,
    desc: str,
    axial_image_norm_coord: dict,
    axial_crop_mm: list[float],
) -> list[dict]:
    data = []
    for i in stack_indices:
        image = stack['arrays'][i]
        image_shape = stack['image_shapes'][i]
        pixel_spacing = stack['pixel_spacings'][i]

        (crop_top, crop_bottom), (crop_left, crop_right) = (np.array(axial_crop_mm) / pixel_spacing).tolist()

        x, y = axial_image_norm_coord['center'][0] * image_shape[1], axial_image_norm_coord['center'][1] * image_shape[0]
        left = round(x - crop_left)
        right = round(x + crop_right)
        top = round(y - crop_top)
        bottom = round(y + crop_bottom)

        crop_image = image[top:bottom, left:right]
        data.append(dict(
            study_id=study_id,
            series_id=series_id,
            series_description=desc,
            level=level,
            instance_no=stack['instance_numbers'][i],
            standard_pos=stack['standard_positions'][i],
            image=crop_image,
            dataset_version=11,
        ))

    return data


print('Start prediction')
st_ids = TEST_DESC_DF['study_id'].unique()
descs = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']
pred_df_list = []

all_level_studies = []
for idx, study_id in enumerate(tqdm(st_ids, total=len(st_ids))):
    data = []
    pdf = TEST_DESC_DF[TEST_DESC_DF['study_id'] == study_id]

    desc_to_series_id = {}
    for desc in descs:
        series_ids = pdf[pdf['series_description'] == desc]['series_id'].values.tolist()
        desc_to_series_id[desc] = series_ids

    # Dicom stackの読み込み
    stacks = {series_id: None for series_ids in desc_to_series_id.values() for series_id in series_ids}
    for desc, series_ids in desc_to_series_id.items():
        if len(series_ids) == 0:
            print(study_id, desc, 'has no images')
            continue

        for series_id in series_ids:
            dicom_folder = f'{IMAGE_ROOT}/{study_id}/{series_id}'
            stack = load_dicom_stack(dicom_folder, desc.split(' ')[0])
            stacks[series_id] = stack

    # Sagittal画像の収集
    sagittal_world_coords = {'Sagittal T1': [], 'Sagittal T2/STIR': []}
    for desc in ['Sagittal T1', 'Sagittal T2/STIR']:
        for series_id in desc_to_series_id[desc]:
            stack = stacks[series_id]
            if stack is None:
                continue

            # 椎間板レベルの位置を検出
            sagittal_image_coord, sagittal_image_norm_coord, sagittal_world_coord = SAGITTAL_KEYPOINT_DETECTOR.predict(stacks[series_id])
            sagittal_world_coords[desc].append(sagittal_world_coord)

            # Dataset v9, v11のデータを作成))
            data.extend(create_dataset_v8_sagittal(stack, study_id, series_id, desc, sagittal_image_norm_coord,
                                                   DATASET_V8_SAGITTAL_CROP_MM, DATASET_V8_SAGITTAL_CROP_OFFSET_MM))

            data.extend(create_dataset_v9_sagittal(stack, study_id, series_id, desc, sagittal_image_norm_coord,
                                                   DATASET_V9_SAGITTAL_CROP_MM, DATASET_V9_SAGITTAL_CROP_OFFSET_MM))

            data.extend(create_dataset_v11_sagittal(stack, study_id, series_id, desc, sagittal_image_norm_coord,
                                                    DATASET_V11_SAGITTAL_CROP_MM, DATASET_V11_SAGITTAL_CROP_OFFSET_MM))

    # Axial画像の収集
    if len(sagittal_world_coords['Sagittal T2/STIR']):
        sagittal_world_coord = sagittal_world_coords['Sagittal T2/STIR'][0]
    else:
        sagittal_world_coord = sagittal_world_coords['Sagittal T1'][0]

    points = extend_sagment_point(sagittal_world_coord)
    lines = {'L1/L2': [points['L1/L2_upper'], points['L1/L2-L2/L3_mid']],
             'L2/L3': [points['L1/L2-L2/L3_mid'], points['L2/L3-L3/L4_mid']],
             'L3/L4': [points['L2/L3-L3/L4_mid'], points['L3/L4-L4/L5_mid']],
             'L4/L5': [points['L3/L4-L4/L5_mid'], points['L4/L5-L5/S1_mid']],
             'L5/S1': [points['L4/L5-L5/S1_mid'], points['L5/S1_lower']]}

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

                axial_image_coord, axial_image_norm_coord, axial_world_coord = AXIAL_KEYPOINT_DETECTOR.predict({
                    'instance_numbers': stack['instance_numbers'][stack_indices],
                    'arrays': [stack['arrays'][i] for i in stack_indices],
                    'image_shapes': stack['image_shapes'][stack_indices],
                    'image_position_patients': stack['image_position_patients'][stack_indices],
                    'image_orientation_patients': stack['image_orientation_patients'][stack_indices],
                    'pixel_spacings': stack['pixel_spacings'][stack_indices],
                    'standard_positions': stack['standard_positions'][stack_indices],
                })

                data.extend(create_dataset_v8_axial(stack, stack_indices, study_id, series_id, level, desc, axial_image_norm_coord,
                                                    DATASET_V8_AXIAL_CROP_MM, DATASET_V8_AXIAL_CROP_OFFSET_MM))

                data.extend(create_dataset_v9_axial(stack, stack_indices, study_id, series_id, level, desc, axial_image_norm_coord,
                                                    DATASET_V9_AXIAL_CROP_MM, DATASET_V9_AXIAL_CROP_OFFSET_MM))

                data.extend(create_dataset_v11_axial(stack, stack_indices, study_id, series_id, level, desc, axial_image_norm_coord,
                                                     DATASET_V11_AXIAL_CROP_MM))

    study_image_df = pd.DataFrame(data=data)

    # 疾患の分類
    pred_df_level_split = LEVEL_SPLIT_DETECTOR.predict(study_id, study_image_df)
    pred_df_level_spinal_split = LEVEL_SPINAL_SPLIT_DETECTOR.predict(study_id, study_image_df)
    pred_df_level_lr_split = LEVEL_LR_SPLIT_DETECTOR.predict(study_id, study_image_df)

    # 脊柱管狭窄強化モデルで脊柱管狭窄を予測する
    pred_df_level_split.insert(1, 'type', 'level_split')
    pred_df_level_split = pred_df_level_split[pred_df_level_split['row_id'].str.contains('spinal')]
    pred_df_list.append(pred_df_level_split)

    # 脊柱管狭窄専用モデルで脊柱管狭窄を予測する
    pred_df_level_spinal_split.insert(1, 'type', 'level_spinal_split')
    pred_df_list.append(pred_df_level_spinal_split)

    # 左右椎間孔狭窄、左右椎間関節下狭窄を予測する
    pred_df_level_lr_split.insert(1, 'type', 'level_lr_split')
    pred_df_list.append(pred_df_level_lr_split)

PRED_DF = pd.concat(pred_df_list, ignore_index=True)

# %% [markdown]
# ## Post processing

# %%
PRED_DF.drop('type', axis=1, inplace=True)
PRED_DF = PRED_DF.groupby('row_id').mean().reset_index()
PRED_DF.sort_values('row_id', inplace=True)

# %% [markdown]
# ## Make submission

# %%
print(f'Save submission to {SUBMISSION_CSV}')
SUBMISSION_CSV.parent.mkdir(exist_ok=True, parents=True)
PRED_DF.to_csv(SUBMISSION_CSV, index=False)
