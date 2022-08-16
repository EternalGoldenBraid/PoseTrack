import os
from time import perf_counter
import sys
import warnings
import json
import cv2
import numpy as np
import h5py
from ipdb import iex

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from ove6d.configs import config as cfg

def main(args):

    grp = args.group
    file = "rs_records.hdf5"

    f = h5py.File(f"data/{file}", "r")
    depth_scale = f['meta']["depth_scale"][:]
    cam_K = f['meta']["camera_intrinsic"][:]
    fps = f['meta']["framerate"][0]

    for dataset in f[grp]:

        color_frames = f[grp][dataset]['color_frames'][:]
        depth_frames = f[grp][dataset]['depth_frames'][:]
        
        n_frames = color_frames.shape[0]
        duration = int(n_frames/fps)
        
        # Streaming loop
        for count in range(n_frames):
        
            depth_image = depth_frames[count]
            color_image = color_frames[count].astype(np.uint8)
        
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
        
            images = np.hstack((color_image, depth_colormap))

            #import pdb; pdb.set_trace()
            
            #cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(int(1000/fps))
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
        
    cv2.destroyAllWindows()
    f.close()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='playback',
            description='Playback recorded frames.')
    
    parser.add_argument('-s','--scenario', dest='group',
                        type=str, default='single_object',
                        choices=['single_object', 'single_object_occluded',
                            'multi_object', 'multi_object_occluded'],
                        help='Scenarios')
    #parser.add_argument('-o','--objects', dest='ds',
    #                    type='string', default='eraser',
    #                    choices=['single_object', 'single_object_occluded',
    #                        'multi_object', 'multi_object_occluded'],
    #                    help='Scenarios')
    parser.add_argument('-d','--duration', dest='duration',
                        type=int, default=10, help='Recording duration')
    parser.add_argument('-fps, --framerate', dest='framerate',
                        type=int, default=30, choices=[6,30,90], help='Recording framerate')

    args = parser.parse_args()
    main(args)