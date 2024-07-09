import numpy as np
# from PIL import Image
# import torch
# from typing import Callable, List, Tuple, Optional
# from sklearn.decomposition import NMF
# from typing import List, Dict
from scipy import ndimage

def scale_cam_map(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = resize_volume(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def resize_volume(img, target_size):
    
    desired_depth = target_size[0]
    desired_width = target_size[1]
    desired_height = target_size[2]

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img