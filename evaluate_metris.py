"""
Calculation of the metrics TPR, TPavg, FPavg, FDR, dice score, IOU. 
"""

import os
import numpy as np
from PIL import Image
from skimage.measure import label 
import pandas as pd
import zarr
from utils import *


def calculate_iou(pred, target):
    """
    Calculate the Intersection over Union (IoU) between predicted and target masks.
    
    Parameters:
    - pred: Predicted binary segmentation map (numpy array)
    - target: Ground truth binary segmentation map (numpy array)
    
    Returns:
    - iou: The IoU score between pred and target
    """
    # Ensure the input maps are numpy arrays and binary
    pred = np.asarray(pred > 0, dtype=np.float32)
    target = np.asarray(target > 0, dtype=np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    # Calculate IoU
    if union == 0:
        # If both pred and target are empty, return 1.0
        # If only one is empty, return 0.0
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou

def calculate_class_wise_iou(pred, target):
    """
    Calculate class-wise IoU for dense, moderate, and light fiber bundles.
    
    Parameters:
    - pred: Predicted segmentation map (numpy array)
    - target: Ground truth segmentation map with classes [1,2,3] (numpy array)
    
    Returns:
    - iou_dense: IoU for dense fibers (class 3)
    - iou_moderate: IoU for moderate fibers (class 2)
    - iou_light: IoU for light fibers (class 1)
    """
    ious = []
    
    # Calculate IoU for each class
    for class_id in [1, 2, 3]:  # light, moderate, dense
        # Create binary masks for the current class
        pred_class = (pred == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)
        
        # Calculate IoU for this class
        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class) - intersection
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
            
        ious.append(iou)
    
    return tuple(ious)  # returns (iou_light, iou_moderate, iou_dense)

def get_clusterwise_metrics_typewise(pred, col_target):
    sens = []
    fp = []
    fn = []
    for ct in [1, 2, 3]:
        labeltarget, num_target = label(col_target == ct, return_num=True)
        tp = np.setdiff1d(np.union1d(labeltarget[pred > 0], []), 0)
        tp = len(list(tp))
        fn = num_target - tp
        if (tp + fn) == 0:
            sens_val = 1
        else:
            sens_val = tp / (tp + fn)
        sens.append(sens_val)
    sens_light, sens_mod, sens_dense = sens

    labelpred, num_pred = label(pred > 0, return_num=True)
    tp = np.setdiff1d(np.union1d(labelpred[col_target > 0], []), 0)
    tp = len(list(tp))
    fp = num_pred - tp
    if num_pred == 0:
        fdr = 0
    else:
        fdr = fp / num_pred

    return sens_dense, sens_mod, sens_light, tp, fp, fdr

def dice_score(pred, target):
    """
    Calculate Dice score for binary classification.
    pred: binary prediction [0,1]
    target: will be converted to binary [0,1] where classes [1,2,3] become 1
    """
    # Ensure binary prediction
    pred_binary = (pred > 0).astype(np.float32)
    
    # Convert multi-class target to binary
    target_binary = (target > 0).astype(np.float32)  # This automatically maps [1,2,3] to 1
    
    # Calculate intersection and union
    intersection = np.sum(pred_binary * target_binary)
    sum_pred = np.sum(pred_binary)
    sum_target = np.sum(target_binary)
    
    # Calculate Dice
    dice = 2 * intersection / (sum_pred + sum_target + 1e-6)
    
    return dice


#########################################################
# Adapt this part according to your data folder structure

subject = '' # choose subject eid

dirname = ''
omz = zarr.open_group(dirname + '', mode='r')
outline = zarr.open_group(dirname + '/masks/Outline_mask.ome.zarr', mode='r')
dense_masks = zarr.open_group(dirname + '/masks/Fiber_dense_bundle.ome.zarr', mode='r')
moderate_masks = zarr.open_group(dirname + '/masks/Fiber_moderate_bundle.ome.zarr', mode='r')
light_masks = zarr.open_group(dirname + '/masks/Fiber_light_bundle.ome.zarr', mode='r')

level = '4'
hist_paths = np.transpose(omz[level], (1, 2, 3, 0))
brain_paths = np.transpose(outline[level], (1, 2, 3, 0))
label_paths_db = np.transpose(dense_masks[level], (1, 2, 3, 0))
label_paths_mb = np.transpose(moderate_masks[level], (1, 2, 3, 0))
label_paths_lb = np.transpose(light_masks[level], (1, 2, 3, 0))

predicted_folder = "/evaluate_results/unet_ensemble/subject/"
results_dir = predicted_folder

slides = [f"{i:03d}" for i in range(1, 36)] #change according to your number of slides
#########################################################
results = []

for ix, slide in enumerate(slides):
    print(ix, slide)
    index = ix

    image, boundaries = tight_crop_data(hist_paths[index])
    mask = crop_mask_like_data(brain_paths[index], boundaries)[:,:,0]

    # Get masks
    target_db = crop_mask_like_data(label_paths_db[index], boundaries)[:,:,0]
    target_mb = crop_mask_like_data(label_paths_mb[index], boundaries)[:,:,0]
    target_lb = crop_mask_like_data(label_paths_lb[index], boundaries)[:,:,0]

    predicted = Image.open(os.path.join(predicted_folder, f'bundle_{slide}_0000_pred.png'))
    predicted_np = np.array(predicted)

    # Normalize to binary (0,1)
    predicted_np = (predicted_np > 127).astype(np.uint8) 

    target_map = np.zeros_like(target_db)
    target_map[target_lb > 0] = 1
    target_map[target_mb > 0] = 2
    target_map[target_db > 0] = 3
    target_map = target_map.astype(int)
    
    # Apply the mask to consider only the region of interest (ROI)
    target = target_map * mask
    pred = predicted_np * mask

    # Compute cluster-wise metrics
    sensds, sensms, sensls, tps, fps, fdrs = get_clusterwise_metrics_typewise(pred, target)

    score = dice_score(pred, target)

    iou_score = calculate_iou(pred, target)
    iou_light, iou_moderate, iou_dense = calculate_class_wise_iou(pred, target)
         
    result = {
            'file': slide,
            'sensitivity_dense': sensds,
            'sensitivity_moderate': sensms,
            'sensitivity_light': sensls,
            'false_positives_vaan_fun': fps,
            'true_positives_vaan_fun': tps,
            'fdr': fdrs,
            'dice_score': score,
            'iou_score': iou_score,           # Added IoU score
            'iou_dense': iou_dense,           # Added class-wise IoU
            'iou_moderate': iou_moderate,
            'iou_light': iou_light
        }
    results.append(result)

# Save the results to a CSV file
df = pd.DataFrame(results)
df.to_csv(os.path.join(results_dir, "bundle_evaluation_results_inside_mask.csv"), index=False)