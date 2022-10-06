import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.random import default_rng
import h5py


SEED = 3333

def load_recorded_frames(filename, object_name, data_path=None):
    if data_path==None:
        data_path = Path('data','recordings',filename)
    else:
        raise NotImplementedError("data_path unsupported")

    with h5py.File(data_path, "r") as f:
        data = f[f'single_object/{object_name}']
        colors = data['color_frames'][:].astype(np.uint8)
        depths = data['depth_frames'][:]
        depth_scale = f['meta/depth_scale'][:]
        cam_K = f['meta/camera_intrinsic'][:]
        fps = f['meta/framerate'][:]
        
    return colors, depths, depth_scale, cam_K, int(fps)

def annotate_images(images, img_size: Tuple[int, int] = (480, 640), 
                    titles: List[List[str]] = [["Title"]]) -> None:

    assert images.flags['C_CONTIGUOUS'] == True
    
    n_rows = images.shape[0]//img_size[0]
    n_cols = images.shape[1]//img_size[1]
    
    for row in range(n_rows):
        for col in range(n_cols):
            images = cv2.putText(img=images, text=titles[row][col], org=(2+col*img_size[1], 20+row*img_size[0]), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5,
                        color=(0,255,0), thickness=1);

    return images
