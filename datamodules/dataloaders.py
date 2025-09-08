import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from datamodules.datasets import HistFinetuneDatasetCachedForeground

class HistFinetuneDataModuleForegroundAware(pl.LightningDataModule):
    def __init__(self,
                 cv_image_dir,
                 cv_label_dir,
                 patch_h=1024,
                 patch_w=1024,
                 num_random_patches=10,
                 foreground_oversample_ratio=0.5,
                 batch_size=4,
                 num_workers=4,
                 transform=None,
                 train_slices=None,
                 val_slices=None):
        super().__init__()
        self.cv_image_dir = Path(cv_image_dir)
        self.cv_label_dir = Path(cv_label_dir)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_random_patches = num_random_patches
        self.foreground_oversample_ratio = foreground_oversample_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.train_slices_override = train_slices
        self.val_slices_override = val_slices

    def setup(self, stage=None):
        # ---- CV Data ----
        image_files = sorted(self.cv_image_dir.glob("*.png"))
        label_files = sorted(self.cv_label_dir.glob("*.png"))
        slice_samples = [{'image_path': i, 'label_path': l} for i, l in zip(image_files, label_files)]

        if self.train_slices_override is not None and self.val_slices_override is not None:
            self.train_slices = self.train_slices_override
            self.val_slices = self.val_slices_override
        else:
            val_split = int(0.8 * len(slice_samples))
            self.train_slices = slice_samples[:val_split]
            self.val_slices = slice_samples[val_split:]

        self.train_dataset = HistFinetuneDatasetCachedForeground(
            self.train_slices, self.patch_h, self.patch_w, self.num_random_patches, self.transform)

        self.val_dataset = HistFinetuneDatasetCachedForeground(
            self.val_slices, self.patch_h, self.patch_w, self.num_random_patches)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers) 
    
