
import numpy as np 
from typing import List, Tuple, Union, Optional

def cut_zeros1d(im_array: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the window for cropping the data closer to the brain.

    Args:
        im_array: Input array to analyze

    Returns:
        Tuple containing:
            - start_index: Starting index of non-zero values
            - end_index: End index of non-zero values
            - length: Length of non-zero intensity values
    """
    im_list = list(im_array > 0)
    if 1 not in im_list:  # Check if there are any non-zero values
        return 0, 0, 0
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:]) - end_index
    return start_index, end_index, length

def tight_crop_data(img_data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Crop the data tighter to the brain.

    Args:
        img_data: Input array

    Returns:
        Tuple containing:
            - cropped_data: The cropped image array
            - List of coordinates [row_start, row_length, col_start, col_length, stack_start, stack_length]
    """
    row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
    col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
    stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)

    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)

    cropped_data = img_data[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen]
    return cropped_data, [rsid, rlen, csid, clen, ssid, slen]

def crop_mask_like_data(mask: np.ndarray, bounding_box: List[int]) -> np.ndarray:
    """
    Crop the masks to match the dimensions of the data using a bounding box.

    Args:
        mask: Input mask array to be cropped
        bounding_box: List containing [row_start, row_length, col_start, col_length, stack_start, stack_length]

    Returns:
        np.ndarray: Cropped mask matching the data dimensions
    """
    rsid, rlen, csid, clen, ssid, slen = bounding_box       
    cropped_mask = mask[rsid:rsid+rlen, csid:csid+clen]
    return cropped_mask

