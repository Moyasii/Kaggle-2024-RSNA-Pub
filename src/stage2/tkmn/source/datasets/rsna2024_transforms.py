from source.datasets.dataset_phase import DatasetPhase
import albumentations as A


import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())


def build_transforms(
    phase: DatasetPhase,
    image_size: tuple[int, int],
    brightness_limit: tuple[float, float],
    contrast_limit: tuple[float, float],
    brightness_contrast_p: float,
    blur_limit: int,
    blur_p: float,
    noise_var_limit: tuple[float, float],
    distort_limit: float,
    grid_distort_limit: float,
    elastic_alpha: float,
    distortion_p: float,
    shift_limit: float,
    scale_limit: float,
    rotate_limit: int,
    shift_scale_rotate_p: float,
    max_holes: int,
    max_height: int,
    max_width: int,
    min_holes: int,
    min_height: int,
    min_width: int,
    coarse_dropout_p: float
) -> A.Compose:
    if phase == DatasetPhase.TRAIN:
        transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=brightness_contrast_p),
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.MedianBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
                A.GaussNoise(var_limit=noise_var_limit),
            ], p=blur_p),

            A.OneOf([
                A.OpticalDistortion(distort_limit=distort_limit),
                A.GridDistortion(num_steps=5, distort_limit=grid_distort_limit),
                A.ElasticTransform(alpha=elastic_alpha),
            ], p=distortion_p),

            A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, border_mode=0, p=shift_scale_rotate_p),
            A.Resize(*image_size),
            A.CoarseDropout(max_holes=max_holes, max_height=max_height, max_width=max_width,
                            min_holes=min_holes, min_height=min_height, min_width=min_width, p=coarse_dropout_p),
            A.Normalize(mean=0.5, std=0.5)
        ])
    else:
        transforms = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=0.5, std=0.5)
        ])
    return transforms
