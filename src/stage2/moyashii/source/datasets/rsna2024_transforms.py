import albumentations as A

import numpy as np
import torch
from torch.distributions import Beta

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.datasets.dataset_phase import DatasetPhase  # noqa


class Mixup:
    def __init__(self, p: float, mix_beta: float = 1.0):
        self.p = p
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def __call__(
        self,
        sagittal_t1: torch.Tensor,
        sagittal_t2: torch.Tensor,
        axial_t2: torch.Tensor,
        targets: torch.Tensor,
        slice_labels: torch.Tensor,
        slice_weights: torch.Tensor,
        level_targets: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        if np.random.random() <= self.p:
            bs = targets.shape[0]
            perm = torch.randperm(bs)
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(sagittal_t1.device)

            sagittal_t1 = coeffs.view(-1, 1, 1, 1) * sagittal_t1 + (1 - coeffs.view(-1, 1, 1, 1)) * sagittal_t1[perm]
            sagittal_t2 = coeffs.view(-1, 1, 1, 1) * sagittal_t2 + (1 - coeffs.view(-1, 1, 1, 1)) * sagittal_t2[perm]
            axial_t2 = coeffs.view(-1, 1, 1, 1) * axial_t2 + (1 - coeffs.view(-1, 1, 1, 1)) * axial_t2[perm]
            targets = coeffs.view(-1, 1, 1, 1) * targets + (1 - coeffs.view(-1, 1, 1, 1)) * targets[perm]
            slice_labels = coeffs.view(-1, 1, 1, 1) * slice_labels + (1 - coeffs.view(-1, 1, 1, 1)) * slice_labels[perm]
            slice_weights = coeffs.view(-1, 1, 1) * slice_weights + (1 - coeffs.view(-1, 1, 1)) * slice_weights[perm]
            level_targets = coeffs.view(-1, 1, 1) * level_targets + (1 - coeffs.view(-1, 1, 1)) * level_targets[perm]

        return sagittal_t1, sagittal_t2, axial_t2, targets, slice_labels, slice_weights, level_targets


def build_transforms(
    phase: DatasetPhase,
    image_size: tuple[int, int],
    brightness_limit: tuple[float, float] = (-0.2, 0.2),
    contrast_limit: tuple[float, float] = (-0.2, 0.2),
    brightness_contrast_p: float = 0.5,
    blur_limit: int = 5,
    gauss_noise_var_limit: tuple[float, float] = (5.0, 30.0),
    blur_p: float = 0.0,
    distort_limit: float = 1.0,
    grid_distort_num_steps: int = 5,
    grid_distort_distort_limit: float = 1.0,
    elastic_transform_alpha: float = 3.0,
    distortion_p: float = 0.5,
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    rotate_limit: int = 8,
    shift_scale_rotate_border_mode: int = 0,
    shift_scale_rotate_p: float = 0.5,
    coarse_dropout_max_holes: int = 16,
    coarse_dropout_max_height: int = 40,
    coarse_dropout_max_width: int = 40,
    coarse_dropout_min_holes: int = 1,
    coarse_dropout_min_height: int = 4,
    coarse_dropout_min_width: int = 4,
    coarse_dropout_p: float = 0.5,
) -> A.Compose:
    if phase == DatasetPhase.TRAIN:
        transforms = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_p
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.MedianBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
                A.GaussNoise(var_limit=gauss_noise_var_limit),
            ], p=blur_p),
            A.OneOf([
                A.OpticalDistortion(distort_limit=distort_limit),
                A.GridDistortion(num_steps=grid_distort_num_steps, distort_limit=grid_distort_distort_limit),
                A.ElasticTransform(alpha=elastic_transform_alpha),
            ], p=distortion_p),
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=shift_scale_rotate_border_mode,
                p=shift_scale_rotate_p
            ),
            A.Resize(*image_size),
            A.CoarseDropout(
                max_holes=coarse_dropout_max_holes,
                max_height=coarse_dropout_max_height,
                max_width=coarse_dropout_max_width,
                min_holes=coarse_dropout_min_holes,
                min_height=coarse_dropout_min_height,
                min_width=coarse_dropout_min_width,
                p=coarse_dropout_p
            ),
            A.Normalize(mean=0.5, std=0.5)
        ])
    else:
        transforms = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=0.5, std=0.5)
        ])
    return transforms


if __name__ == '__main__':
    import torch.nn.functional as F  # noqa
    sagittal_t1 = torch.ones(4, 15, 64, 64)
    sagittal_t2 = torch.zeros(4, 15, 64, 64)
    axial_t2 = torch.zeros(4, 10, 64, 64)
    targets = F.one_hot(torch.randint(0, 3, (4, 5, 5)), num_classes=3).to(torch.float)
    level_targets = F.one_hot(torch.randint(0, 3, (4, 5)), num_classes=5).to(torch.float)

    print(targets)
    mixup = Mixup(1.0)
    sagittal_t1, sagittal_t2, axial_t2, targets, level_targets = mixup(sagittal_t1, sagittal_t2, axial_t2, targets, level_targets)
    print(targets)
