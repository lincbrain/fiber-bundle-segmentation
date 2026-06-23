# Fully Automated Segmentation of Fiber Bundles in Anatomic Tracing Data

## About
This is a Pytorch implementation for the paper "Fully Automated Segmentation of Fiber Bundles in Anatomic Tracing Data" (CDMRI workshop, MICCAI 2025) by Kyriaki-Margarita Bintsi, Yaël Balbastre, Jingjing Wu, Julia F. Lehman, Suzanne N. Haber, and Anastasia Yendiki


---

## Installation
Clone the repository:
```
git clone https://github.com/lincbrain/fiber-bundle-segmentation.git
cd your-repo
```

## Requirements
Install dependencies:
```
pip install -r requirements.txt
```

## Training

### Pretraining
Pretrain the U-Net model for self-supervised reconstruction:
```
python pretrain.py \
    --batch_size 16 \
    --epochs 50 \
    --patch_h 1024 \
    --patch_w 1024 \
    --num_random_patches 20 \
    --dirpath ./pretraining_saved_models/
```

### Fine-tuning
Fine-tune the U-Net model using cross-validation and optional pre-trained weights:
```
python train_finetune.py \
    --batch_size 8 \
    --epochs 1000 \
    --patch_h 1024 \
    --patch_w 1024 \
    --num_random_patches 20 \
    --checkpoint_dir ./finetune_checkpoints/ \
    --pretrained_checkpoint ./pretraining_saved_models/unet_pretraining.ckpt \
    --loss BCEdice
```

### Prediction
Run inference on new images with optional test-time augmentation and small-object removal:
```
python predict.py \
    --input_folder ./test_images/ \
    --output_folder ./predictions/ \
    --model_folder ./finetune_checkpoints/ \
    --saved_model best \
    --min_size 20
```

## Reference
If you find the code useful, pleace cite: 
```
@inproceedings{bintsi2025fully,
  title={Fully Automated Segmentation of Fiber Bundles in Anatomic Tracing Data},
  author={Bintsi, Kyriaki-Margarita and Balbastre, Ya{\"e}l and Wu, Jingjing and Lehman, Julia F and Haber, Suzanne N and Yendiki, Anastasia},
  booktitle={International Workshop on Computational Diffusion MRI},
  pages={81--92},
  year={2025},
  organization={Springer}
}
```
