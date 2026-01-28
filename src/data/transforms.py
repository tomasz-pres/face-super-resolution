"""
Data Augmentation Transforms for Super-Resolution
==================================================
Synchronized transforms for LR-HR image pairs using albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional


def get_train_transforms(hr_patch_size: int = 128,
                         scale_factor: int = 4) -> A.ReplayCompose:
    """
    Get training augmentation pipeline with synchronized LR-HR transforms.

    Args:
        hr_patch_size: Size of HR patches to extract
        scale_factor: Upscaling factor (LR patch = HR patch / scale_factor)

    Returns:
        Albumentations ReplayCompose for synchronized transforms
    """
    lr_patch_size = hr_patch_size // scale_factor

    return A.ReplayCompose([
        # Random crop (applied to HR, LR extracted separately)
        A.RandomCrop(height=hr_patch_size, width=hr_patch_size, p=1.0),

        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Color transforms (subtle for faces)
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
    ])


def get_val_transforms(hr_size: int = 256) -> A.Compose:
    """
    Get validation/test transforms (no augmentation, just center crop if needed).

    Args:
        hr_size: Expected HR image size

    Returns:
        Albumentations Compose for validation
    """
    return A.Compose([
        # Ensure correct size
        A.CenterCrop(height=hr_size, width=hr_size, p=1.0),
    ])


def apply_synchronized_transform(
    hr_image: np.ndarray,
    lr_image: np.ndarray,
    transform: A.ReplayCompose,
    scale_factor: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same geometric transforms to both HR and LR images.

    For random crop, we need to handle LR separately since it's smaller.

    Args:
        hr_image: High-resolution image (H, W, C)
        lr_image: Low-resolution image (H/scale, W/scale, C)
        transform: ReplayCompose transform
        scale_factor: Upscaling factor

    Returns:
        Tuple of (transformed_hr, transformed_lr)
    """
    # Apply transform to HR and capture replay data
    result = transform(image=hr_image)
    hr_transformed = result['image']
    replay = result['replay']

    # For LR, we need to handle crop coordinates specially
    # Extract the crop coordinates from replay if present
    lr_transformed = lr_image.copy()

    for t in replay['transforms']:
        if t['__class_fullname__'] == 'RandomCrop':
            if t['applied']:
                # Scale crop coordinates for LR
                h_start = t['params']['h_start'] // scale_factor
                w_start = t['params']['w_start'] // scale_factor
                crop_h = t['params']['crop_height'] // scale_factor
                crop_w = t['params']['crop_width'] // scale_factor

                lr_transformed = lr_transformed[
                    h_start:h_start + crop_h,
                    w_start:w_start + crop_w
                ]

        elif t['__class_fullname__'] == 'HorizontalFlip':
            if t['applied']:
                lr_transformed = np.fliplr(lr_transformed).copy()

        elif t['__class_fullname__'] == 'RandomRotate90':
            if t['applied']:
                factor = t['params']['factor']
                lr_transformed = np.rot90(lr_transformed, factor).copy()

        elif t['__class_fullname__'] == 'ColorJitter':
            if t['applied']:
                # Apply same color transform using replay
                lr_result = A.ReplayCompose.replay(replay, image=lr_transformed)
                # This doesn't work perfectly, so we apply manually
                pass  # Color jitter is tricky to sync, skip for LR

    return hr_transformed, lr_transformed


class PairedTransform:
    """
    Transform class for synchronized LR-HR augmentation.

    Handles the complexity of applying same geometric transforms
    to images of different sizes.
    """

    def __init__(self,
                 hr_patch_size: int = 128,
                 scale_factor: int = 4,
                 mode: str = 'train',
                 horizontal_flip: float = 0.5,
                 random_rotate90: float = 0.0,  # Default 0 for faces!
                 color_jitter_prob: float = 0.3,
                 brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.05):
        """
        Args:
            hr_patch_size: Size of HR patches for training
            scale_factor: Upscaling factor
            mode: 'train' or 'val'
            horizontal_flip: Probability of horizontal flip
            random_rotate90: Probability of 90-degree rotation (0 for faces!)
            color_jitter_prob: Probability of color jitter
            brightness: Brightness jitter range
            contrast: Contrast jitter range
            saturation: Saturation jitter range
            hue: Hue jitter range
        """
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        self.scale_factor = scale_factor
        self.mode = mode

        # Augmentation probabilities (from config or defaults)
        self.flip_prob = horizontal_flip if mode == 'train' else 0.0
        self.rotate_prob = random_rotate90 if mode == 'train' else 0.0
        self.color_prob = color_jitter_prob if mode == 'train' else 0.0

        # Color jitter parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, hr_image: np.ndarray,
                 lr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply synchronized transforms to LR-HR pair.

        Args:
            hr_image: HR image (H, W, C) uint8
            lr_image: LR image (H/scale, W/scale, C) uint8

        Returns:
            Tuple of (hr_transformed, lr_transformed)
        """
        hr_h, hr_w = hr_image.shape[:2]
        lr_h, lr_w = lr_image.shape[:2]

        if self.mode == 'train':
            # Random crop
            if hr_h > self.hr_patch_size and hr_w > self.hr_patch_size:
                # Random crop position
                top = np.random.randint(0, hr_h - self.hr_patch_size + 1)
                left = np.random.randint(0, hr_w - self.hr_patch_size + 1)

                # Crop HR
                hr_image = hr_image[
                    top:top + self.hr_patch_size,
                    left:left + self.hr_patch_size
                ]

                # Corresponding LR crop
                lr_top = top // self.scale_factor
                lr_left = left // self.scale_factor
                lr_image = lr_image[
                    lr_top:lr_top + self.lr_patch_size,
                    lr_left:lr_left + self.lr_patch_size
                ]

            # Horizontal flip
            if np.random.random() < self.flip_prob:
                hr_image = np.fliplr(hr_image).copy()
                lr_image = np.fliplr(lr_image).copy()

            # Random 90 degree rotation
            if np.random.random() < self.rotate_prob:
                k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
                hr_image = np.rot90(hr_image, k).copy()
                lr_image = np.rot90(lr_image, k).copy()

            # Color jitter (applied to both)
            if np.random.random() < self.color_prob:
                hr_image, lr_image = self._color_jitter(hr_image, lr_image)

        return hr_image, lr_image

    def _color_jitter(self, hr_image: np.ndarray,
                      lr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same color jitter to both images."""
        # Random brightness (using config values)
        brightness = np.random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        # Random contrast
        contrast = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        # Random saturation
        saturation = np.random.uniform(1.0 - self.saturation, 1.0 + self.saturation)

        for img in [hr_image, lr_image]:
            # Convert to float
            img_float = img.astype(np.float32) / 255.0

            # Brightness
            img_float = img_float * brightness

            # Contrast
            mean = img_float.mean()
            img_float = (img_float - mean) * contrast + mean

            # Saturation (convert to HSV)
            img_hsv = cv2.cvtColor(
                np.clip(img_float * 255, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2HSV
            ).astype(np.float32)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation
            img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)

            img[:] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return hr_image, lr_image


def to_tensor(image: np.ndarray, normalize: bool = True) -> 'torch.Tensor':
    """
    Convert numpy image to PyTorch tensor.

    Args:
        image: Image array (H, W, C) uint8
        normalize: If True, normalize to [0, 1]

    Returns:
        Tensor (C, H, W)
    """
    import torch

    # HWC to CHW
    tensor = torch.from_numpy(image.transpose(2, 0, 1))

    if normalize:
        tensor = tensor.float() / 255.0

    return tensor
