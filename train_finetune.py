import argparse
import os
from pathlib import Path
import wandb

import torch
torch.set_float32_matmul_precision('high') 

from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from datamodules.dataloaders import HistFinetuneDataModuleForegroundAware
from sklearn.model_selection import KFold
from unet import *
import albumentations as A

print(torch.cuda.is_available())

def get_args_parser():
    parser = argparse.ArgumentParser('U-Net finetuning-segmentation', add_help=False)

    # paths parameters
    parser.add_argument('--cv_image_dir', default='../imagesTr/', type=str, help='Folder containing images')
    parser.add_argument('--cv_label_dir', default='../labelsTr/', type=str, help='Folder containing labels')

    # common parameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=1000, type=int)

    parser.add_argument('--patch_h', type=int, default=1024, help='Patch height for training')
    parser.add_argument('--patch_w', type=int, default=1024, help='Patch width for training')
    parser.add_argument('--num_random_patches', type=int, default=20, help='Number of random patches for training')
    
    parser.add_argument('--checkpoint_dir', type=str,help="Directory to save checkpoints")
    parser.add_argument('--pretrained_checkpoint', type=str, help="Path to pretrained model's checkpoint")
    
    parser.add_argument('--model', default='unet', help='Model used for the training')
    parser.add_argument('--loss', default='BCEdice', help='Loss function used for the training')

    # other parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')   
    parser.set_defaults(pin_mem=True)

    return parser 

def main(args):

    print('Experiment saved here', args.checkpoint_dir)

    # Map fold number to checkpoint version
    checkpoint_versions = {
        0: "last.ckpt",      # First fold uses last.ckpt
        1: "last-v1.ckpt",   # Second fold uses last-v1.ckpt
        2: "last-v2.ckpt",   # Third fold uses last-v2.ckpt
        3: "last-v3.ckpt",   # Fourth fold uses last-v3.ckpt
        4: "last-v4.ckpt"    # Fifth fold uses last-v4.ckpt
    }

    # Fixed random seeds
    seed_everything(args.seed, workers=True)

    # Defining data augmentation
    transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(p=0.1),
])
    # Load dataset
    cv_image_dir = args.cv_image_dir
    cv_label_dir = args.cv_label_dir
    img_dir = Path(cv_image_dir)
    lbl_dir = Path(cv_label_dir)
    assert img_dir.exists(), f"Image directory {img_dir} does not exist."
    assert lbl_dir.exists(), f"Label directory {lbl_dir} does 0not exist."

    # ---- Prepare All Samples ----
    image_files = sorted(img_dir.glob("*.png"))
    label_files = sorted(lbl_dir.glob("*.png"))
    assert len(image_files) == len(label_files), "Mismatch between images and labels."
    all_slices = [{'image_path': i, 'label_path': l} for i, l in zip(image_files, label_files)]

    # ---- 5-Fold Cross-Validation ----
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_slices)):
        print(f"\n🚀 Fold {fold+1}/5")

        train_slices = [all_slices[i] for i in train_idx]
        val_slices = [all_slices[i] for i in val_idx]

        # Get the appropriate checkpoint name for this fold
        checkpoint_name = checkpoint_versions[fold]
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint for fold {fold+1}: {checkpoint_path}")
        else:
            print(f"No checkpoint found for fold {fold+1}, starting from scratch")

        # ------- wandb logging -------
        wandb_logger = WandbLogger(
            project="unet-finetuning-segmentation",
            name=f"unet_fold_{fold+1}_loss_{args.loss}",
            save_code = False,
            log_model=False
        )

        datamodule = HistFinetuneDataModuleForegroundAware(
            cv_image_dir=cv_image_dir,
            cv_label_dir=cv_label_dir,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            num_random_patches=args.num_random_patches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform,
            train_slices = train_slices,
            val_slices = val_slices
        )

        # model = UNet()
        model = FlexibleUNet(loss=args.loss)
        if args.pretrained_checkpoint:  # Reuse checkpoint for UNet pre-trained weights
            pretrained_checkpoint = torch.load(args.pretrained_checkpoint)
            if 'state_dict' in pretrained_checkpoint:
                # Extract only the UNet model weights and remove the 'model.' prefix
                state_dict = {}
                for k, v in pretrained_checkpoint['state_dict'].items():
                    if k.startswith('model.'):
                        # Remove 'model.' prefix to match current architecture
                        new_key = k.replace('model.', '')
                        state_dict[new_key] = v
                # Load the weights
                model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore channel_adapter

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir, # folder where ckpt will be saved
            filename=f"best_fold_{fold+1}" + "-{epoch:02d}-{val_loss:.4f}",  
            monitor = "val_loss",
            mode = "min",
            save_last=True,           
            save_top_k=1       
            )

        # Trainer
        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices = 1,
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback]
        )

        # Fit with the appropriate checkpoint if it exists
        if os.path.exists(checkpoint_path):
            print(f"Resuming training for fold {fold+1} from checkpoint: {checkpoint_path}")
            trainer.fit(model, 
                       datamodule=datamodule,
                       ckpt_path=checkpoint_path)
        else:
            print(f"Starting training for fold {fold+1} from scratch")
            trainer.fit(model, datamodule=datamodule)

        # Finish the current WandB run
        wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()  
    main(args) 