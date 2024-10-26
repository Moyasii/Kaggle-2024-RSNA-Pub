from source.datasets.dataset_phase import DatasetPhase
import albumentations as A


import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())


def build_transforms(
    phase: DatasetPhase,
    image_size: tuple[int, int],
    include_motion_blur: bool = True,
    include_median_blur: bool = True,
    include_gaussian_blur: bool = True,
    include_gauss_noise: bool = True,
    noise_p: float = 0.5,
    random_resized_crop_p: float = 0.0,
    brightness_contrast_params: dict = {'brightness_limit': (-0.2, 0.2), 'contrast_limit': (-0.2, 0.2), 'p': 0.5},
    motion_blur_params: dict = {'blur_limit': 5},
    median_blur_params: dict = {'blur_limit': 5},
    gaussian_blur_params: dict = {'blur_limit': 5},
    gauss_noise_params: dict = {'var_limit': (5.0, 30.0)},
    shift_scale_rotate_params: dict = {'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 8, 'border_mode': 0, 'p': 0.5},
    coarse_dropout_params: dict = {'max_holes': 16, 'max_height': 24, 'max_width': 24, 'min_holes': 1, 'min_height': 8, 'min_width': 8, 'p': 0.5},
) -> A.Compose:
    if phase == DatasetPhase.TRAIN:
        transforms_list = [
            A.RandomBrightnessContrast(**brightness_contrast_params)
        ]

        if any([include_motion_blur, include_median_blur, include_gaussian_blur, include_gauss_noise]):
            one_of_transforms = []
            if include_motion_blur:
                one_of_transforms.append(A.MotionBlur(**motion_blur_params))
            if include_median_blur:
                one_of_transforms.append(A.MedianBlur(**median_blur_params))
            if include_gaussian_blur:
                one_of_transforms.append(A.GaussianBlur(**gaussian_blur_params))
            if include_gauss_noise:
                one_of_transforms.append(A.GaussNoise(**gauss_noise_params))
            transforms_list.append(A.OneOf(one_of_transforms, p=noise_p))

        transforms_list.extend([
            A.OneOf([
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(0.5, 0.6)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(0.6, 0.7)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(0.7, 0.8)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(0.8, 0.9)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.0, 1.1)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.1, 1.2)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.2, 1.3)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.3, 1.4)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.4, 1.5)),
                A.RandomResizedCrop(height=600, width=600, scale=(0.85, 1.0), ratio=(1.5, 1.6)),
            ], p=random_resized_crop_p),
            A.ShiftScaleRotate(**shift_scale_rotate_params),
            A.Resize(*image_size),
            A.CoarseDropout(**coarse_dropout_params),
            A.Normalize(mean=0.5, std=0.5)
        ])

        transforms = A.Compose(
            transforms_list,
            keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)
        )
    else:
        transforms = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=0.5, std=0.5)
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
    print(transforms)
    return transforms
