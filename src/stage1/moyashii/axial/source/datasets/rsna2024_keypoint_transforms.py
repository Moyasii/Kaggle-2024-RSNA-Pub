from source.datasets.dataset_phase import DatasetPhase
import albumentations as A


import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())


def build_transforms(phase: DatasetPhase, image_size: tuple[int, int]) -> A.Compose:
    if phase == DatasetPhase.TRAIN:
        transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, border_mode=0, p=0.5),
            A.Resize(*image_size),
            A.CoarseDropout(max_holes=16, max_height=24, max_width=24, min_holes=1, min_height=8, min_width=8, p=0.5),
            A.Normalize(mean=0.5, std=0.5)
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
    else:
        transforms = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=0.5, std=0.5)
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
    return transforms
