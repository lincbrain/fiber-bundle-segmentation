"""
Script to pre-train the UNet.
Usage:
    python pretrain.py --data_root /path/to/data --dirpath ./checkpoints
"""

import argparse
import numpy as np
from pathlib import Path

import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from datamodules.datasets import HistPretrainDataset
from unet_pretraining import PretrainingModule
from unet import *

print(torch.cuda.is_available())

def get_args_parser():
    parser = argparse.ArgumentParser('U-Net pre-training', add_help=False)

    # dir parameters
    parser.add_argument('--data_root', type=str, required=True, help='Root folder containing directories')

    # common parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--patch_h', type=int, default=1024, help='Patch height for training')
    parser.add_argument('--patch_w', type=int, default=1024, help='Patch width for training')
    parser.add_argument('--num_random_patches', type=int, default=20, help='Number of random patches for training')

    parser.add_argument('--model', default='unet', help='Model used for the training')
    parser.add_argument('--dirpath', default='../pretraining_saved_models/', help='Folder where ckpt will be saved')

    # other parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    # parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers for data loading') #TODO: TRY THIS
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):
    # Enable benchmarking mode for better performance
    torch.backends.cudnn.benchmark = True

    # Fixed random seeds
    seed = args.seed
    seed_everything(seed, workers=True)

    # # Set up training equipment
    device = torch.device(args.device)

    # ------- wandb logging -------
    wandb_logger = WandbLogger(
        project="unet-pretraining",
        name=f"unet_pretraining",
        save_code = False,
        log_model=False
    )

    # Load dataset
    root_dir = Path(args.data_root)
    all_dirs = [str(folder/"micr") for folder in root_dir.iterdir() if (folder/"micr").is_dir()]
    
    train_dirs = all_dirs[:15]
    val_dirs = all_dirs[15:]
    print('Train dirs', train_dirs) 
    print('Val dirs', val_dirs)

    train_dataset = HistPretrainDataset(train_dirs, patch_h=args.patch_h, patch_w=args.patch_w, num_random_patches=args.num_random_patches)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
   
    val_dataset = HistPretrainDataset(val_dirs, patch_h=args.patch_h, patch_w=args.patch_w, num_random_patches=args.num_random_patches)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    mdl = FlexibleUNet()

    checkpoint_callback = ModelCheckpoint(
    dirpath=args.dirpath, # folder where ckpt will be saved
    filename="unet_pretraining",  # name of the file
    save_last=True,             # save only the latest checkpoint
    save_top_k=1,
    monitor='val_loss',
    mode='min'
    )

    # Initialize the pretraining module
    model = PretrainingModule(
    model=mdl,
    learning_rate=1e-4
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices = 1,
        logger=wandb_logger,
        precision=16,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader) 
        
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()  
    main(args)  
