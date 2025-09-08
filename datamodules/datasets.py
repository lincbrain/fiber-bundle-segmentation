from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import random
from typing import List, Tuple, Union, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import zarr
from utils import *

def find_zarr_files(directories: Union[str, List[str]]) -> List[Path]:
    """
    Finds all .ome.zarr files in the given directories.

    Args:
        directories: Single directory path or list of directory paths

    Returns:
        List of Path objects for found .ome.zarr files
    """
    if isinstance(directories, str):
        directories = [directories]
        
    zarr_files = []
    for dirname in directories:
        dirname = Path(dirname)
        files = list(dirname.glob("*.ome.zarr"))
        if not files:
            print(f"Warning: No .ome.zarr files found in {dirname}")
            continue
        zarr_files.extend(files)
    return zarr_files

def extract_random_patches(
    image: np.ndarray,
    patch_h: int = 1024,
    patch_w: int = 1024,
    num_patches: int = 10
) -> List[np.ndarray]:
    """
    Extracts random patches from a large image.

    Args:
        image: Input image of shape (H, W, C)
        patch_h: Height of patches to extract
        patch_w: Width of patches to extract
        num_patches: Number of patches to extract

    Returns:
        List of extracted patches
    
    Raises:
        ValueError: If image dimensions are smaller than patch dimensions
    """
    H, W, C = image.shape
    if H < patch_h or W < patch_w:
        raise ValueError(
            f"Image too small: got H={H}, W={W}, required H>={patch_h}, W>={patch_w}"
        )

    patches = []
    for _ in range(num_patches):
        i = random.randint(0, H - patch_h)
        j = random.randint(0, W - patch_w)
        patch = image[i:i+patch_h, j:j+patch_w, :] #(patch_h, patch_w, C)
        patch = np.ascontiguousarray(patch)
        patches.append(patch)
    
    return patches

def extract_foreground_coords(label: np.ndarray, min_area: int = 0) -> List[Tuple[int, int]]:
    """
    Extract all foreground pixel coordinates.

    Args:
        label: Input label array
        min_area: Minimum area threshold

    Returns:
        List of coordinate tuples for foreground pixels
    """
    coords = np.argwhere(label > 0)
    if coords.shape[0] == 0:
        return []
    return [tuple(coord) for coord in coords]

class HistPretrainDataset(Dataset):
    """
    Dataset for Pretraining.
    
    The data is in the format of stack of 3D slices N x C x H x W.

    Args:
        data_dirs: Path or list of paths to directories containing OME-Zarr files
        patch_h: Height of patches to extract
        patch_w: Width of patches to extract
        num_random_patches: Number of random patches per slice
        transform: Optional transform to be applied on samples
    """
    def __init__(self, data_dirs, patch_h=1024, patch_w=1024, num_random_patches=10, transform=None):
        if isinstance(data_dirs, str):  
            data_dirs = [data_dirs]  # Convert single directory to list
        self.zarr_files = find_zarr_files(data_dirs)  # Store file paths instead of loading
        self.transform = transform
        self.level = '4'  # Define resolution level
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_random_patches = num_random_patches
        self.skip_files = []

    def __len__(self):
        """Total number of 2D patches across all .ome.zarr files."""
        total_slices = 0
        for zarr_file in self.zarr_files:
            if str(zarr_file) in self.skip_files:
                continue
            omz = zarr.open_group(zarr_file, mode='r')  # Open Zarr file
            total_slices += omz['8'].shape[1]  # Sum up slices (N dimension) # I put 8 here to make it faster
        return total_slices* self.num_random_patches
    
    def normalize_image(self, img):
        """
        Normalize image using Z-score normalization per channel.

        Args:
            img: Input image array (H, W, C)

        Returns:
            Tuple containing:
                - Normalized image array
                - List of channel means
                - List of channel standard deviations
        """
        means = []
        stds = []
        img = img.astype(np.float32)
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            std = std if std != 0 else 1.0
            means.append(mean)
            stds.append(std)
            img[:, :, c] = (channel - mean) / std
        return img, means, stds
    
    def __getitem__(self, idx):        
        """Load an image slice and return a random patch."""
        cumulative_slices = 0
        for zarr_file in self.zarr_files:
            if str(zarr_file) in self.skip_files:
                continue

            omz = zarr.open_group(zarr_file, mode='r')
            num_slices = omz[self.level].shape[1]  # Get N (number of slices)
            total_patches = num_slices * self.num_random_patches

            if idx < cumulative_slices + total_patches:
                # Find the specific slice and patch we need
                slice_idx = (idx - cumulative_slices) // self.num_random_patches  # Determine slice index
                patch_idx = (idx - cumulative_slices) % self.num_random_patches  # Determine patch index

                # Load the specific slice (N, C, H, W) in Zarr -> (H, W, C) numpy
                img = np.transpose(omz[self.level][:, slice_idx, :, :], (1, 2, 0))
                
                # Apply cropping
                img, _ = tight_crop_data(img)  # (H, W, C)

                if img.shape == (0,0,0):
                    print('Only zeros?', zarr_file, slice_idx)
                    return self.__getitem__((idx + 1) % len(self))
                
                 # # Normalize (For Norm 0-1 uncomment the /255.0)
                img = img.astype(np.float32) #/ 255.0 
                
                # Check size and skip if too small
                H, W, _ = img.shape
                if H < self.patch_h or W < self.patch_w:
                    print('Image too small? Shapes:', H, W, zarr_file, slice_idx)
                    return self.__getitem__((idx + 1) % len(self))  # Try next sample safely
            
                # Normalize before extracting patches - ZScoreNorm! - statistics per image 
                img, means, stds = self.normalize_image(img)
                
                # Extract all random patches
                patches = extract_random_patches(img, self.patch_h, self.patch_w, self.num_random_patches)

                # Return the specific patch
                patch = patches[patch_idx]

                patch_tensor = torch.tensor(np.array(patch)).permute(2, 0, 1).contiguous()  # (C, H, W)
                means_tensor = torch.tensor(means, dtype=torch.float32).contiguous() 
                stds_tensor = torch.tensor(stds, dtype=torch.float32).contiguous() 

                assert isinstance(patch_tensor, torch.Tensor)
                assert patch_tensor.is_contiguous()
                assert patch_tensor.dtype == torch.float32 

                assert isinstance(means_tensor, torch.Tensor)
                assert means_tensor.is_contiguous()
                assert means_tensor.dtype == torch.float32

                assert isinstance(stds_tensor, torch.Tensor)
                assert stds_tensor.is_contiguous()
                assert stds_tensor.dtype == torch.float32

                if not (isinstance(patch_tensor, torch.Tensor) and
                        isinstance(means_tensor, torch.Tensor) and
                        isinstance(stds_tensor, torch.Tensor)):
                    print("BAD TYPE", type(patch_tensor), type(means_tensor), type(stds_tensor))
                    raise RuntimeError("Non-tensor output!")

                if patch_tensor.shape != (3, 1024, 1024):
                    print(f"BAD PATCH SHAPE: {patch_tensor.shape} at idx={idx}")
                    raise RuntimeError("Bad patch shape!")

                if means_tensor.shape != (3,) or stds_tensor.shape != (3,):
                    print(f"BAD MEAN/STD SHAPE: means {means_tensor.shape}, stds {stds_tensor.shape} at idx={idx}")
                    raise RuntimeError("Bad mean/std shape!")

                return patch_tensor, means_tensor, stds_tensor

            cumulative_slices += total_patches  # Move to the next range of patches

        raise IndexError("Index out of range")  # Shouldn't happen if `__len__()` is correct

class HistFinetuneDatasetCachedForeground(Dataset):
    """Dataset for fine-tuning with cached foreground locations.
    
    This dataset handles image segmentation tasks with optional foreground oversampling,
    which can be useful when dealing with imbalanced data where foreground pixels are rare.
    
    Args:
        slice_samples (List[Dict]): List of sample dictionaries containing paths and metadata
        patch_h (int, optional): Height of patches to extract. Defaults to 1024.
        patch_w (int, optional): Width of patches to extract. Defaults to 1024.
        num_random_patches (int, optional): Number of patches per sample. Defaults to 10.
        transform (callable, optional): Optional transform to be applied on patches.
        foreground_oversample_ratio (float, optional): Ratio of patches centered on foreground.
            Should be between 0 and 1. Defaults to 0.5.
            
    Attributes:
        channel_means (List[float]): Per-channel mean values for normalization
        channel_stds (List[float]): Per-channel standard deviation values for normalization
    """
    def __init__(self, slice_samples, patch_h=1024, patch_w=1024,
                 num_random_patches=10, transform=None, foreground_oversample_ratio=0.5): 
        self.slice_samples = slice_samples
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_random_patches = num_random_patches
        self.transform = transform
        self.foreground_oversample_ratio = foreground_oversample_ratio

        # Use the exact statistics
        self.channel_means = [
            26.370181406062926,  # channel 0
            28.796625947529606,  # channel 1
            25.990918044351993   # channel 2
        ]
        self.channel_stds = [
            31.776848516250062,  # channel 0
            28.548426642727726,  # channel 1
            16.703267170423146   # channel 2
        ]
        
        print(f"Using dataset statistics:")
        print(f"Means: {self.channel_means}")
        print(f"Stds: {self.channel_stds}")

        self.flat_samples = []
        for sample in slice_samples:
            # Load the label image to extract foreground locations
            label = np.array(Image.open(sample['label_path']).convert("L")).astype(np.int64)
            fg_coords = extract_foreground_coords(label)

            # Add to metadata
            sample['fg_coords'] = fg_coords
            sample['label_shape'] = label.shape  # used for fallback cropping

            for _ in range(num_random_patches):
                self.flat_samples.append(sample)

    def normalize_image(self, image):
        """
        Apply Z-score normalization using dataset statistics
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            normalized[:, :, c] = (image[:, :, c] - self.channel_means[c]) / self.channel_stds[c]
        return normalized
    
    def crop_patch_centered_at(self, image, label, center_i, center_j):
        H, W = label.shape
        half_h = self.patch_h // 2
        half_w = self.patch_w // 2

        # Ensure the crop does not exceed image boundaries
        top = max(0, min(H - self.patch_h, center_i - half_h))
        left = max(0, min(W - self.patch_w, center_j - half_w))

        img_patch = image[top:top + self.patch_h, left:left + self.patch_w, :]
        mask_patch = label[top:top + self.patch_h, left:left + self.patch_w]
        return img_patch, mask_patch

    def get_random_patch(self, image, label):
        H, W = label.shape
        i = random.randint(0, H - self.patch_h)
        j = random.randint(0, W - self.patch_w)
        img_patch = image[i:i+self.patch_h, j:j+self.patch_w, :]
        mask_patch = label[i:i+self.patch_h, j:j+self.patch_w]
        return img_patch, mask_patch

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        image = np.array(Image.open(sample['image_path']).convert("RGB")).astype(np.float32)  #/ 255.0
        label = np.array(Image.open(sample['label_path']).convert("L")).astype(np.int64)

        use_fg = random.random() < self.foreground_oversample_ratio
        fg_coords = sample['fg_coords']

        if use_fg and fg_coords:
            center_i, center_j = random.choice(fg_coords)
            img_patch, mask_patch = self.crop_patch_centered_at(image, label, center_i, center_j)
        else:
            img_patch, mask_patch = self.get_random_patch(image, label)

        # Apply global Z-score normalization
        img_patch = self.normalize_image(img_patch)

        if self.transform:
            transformed = self.transform(image=img_patch, mask=mask_patch)
            img_patch = transformed['image']
            mask_patch = transformed['mask']

        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1)  # (C, H, W)
        mask_patch = torch.from_numpy(mask_patch)

        return img_patch, mask_patch

    def __len__(self):
        return len(self.flat_samples)
