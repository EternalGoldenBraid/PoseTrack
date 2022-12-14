import os
from pathlib import Path
from time import perf_counter
import sys
import warnings
import json
import cv2
import numpy as np
import h5py
from ipdb import iex
from tqdm import tqdm

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from ove6d.configs import config as cfg

def main(args):

    grp = args.group
    file = f"{args.file}.hdf5"

    f = h5py.File(f"data/{file}", "r")
    depth_scale = f['meta']["depth_scale"][:]
    cam_K = f['meta']["camera_intrinsic"][:]

    if args.framerate == 0:
        fps = f['meta']["framerate"][0]
    else:
        fps = args.framerate

    print("Playback data:", file, grp, " at framerate:", fps)

    for object_recording in f[grp]:
        print("Recorded object:", object_recording)


        # Load object frames
        obj_path = Path(f.name, grp, object_recording)
        rendered_frames = False
        rm = ['color_frames', 'depth_frames']
        cad_list = [i for i in f[str(obj_path)].keys() if i not in rm]
        import pdb; pdb.set_trace()
        #if 'rendered_frames' in f[str(obj_path)].keys():
        if len(cad_list) > 0:
            # Rendered frames exist for some cad model(s).
            rendered_frames = True
            # Combine rendered cad models to single window.
            #import pdb; pdb.set_trace()
            frames = np.concatenate(np.array([f[str(obj_path/cad_model/'rendered_frames')] for cad_model in cad_list]) ,axis=2)
            #frames = f[str(obj_path)]['rendered_frames'][:][None, ...]
            #poses = f[str(obj_path/'poses')][:]
            n_frames = frames.shape[0]
        else:
            color_frames = f[str(obj_path/'color_frames')][:]
            depth_frames = f[str(obj_path/'depth_frames')][:]
            #frames = np.stack((color_frames, depth_frames), axis=0)
            n_frames = color_frames.shape[0]

        duration = int(n_frames/fps)
        
        # Playback frames
        print("Playing n_frames:", n_frames, ",duration:", duration, ",fps:",fps)
        for count in range(n_frames):
            
            if rendered_frames:
                images = frames[count].astype(np.uint8)
                #print("Pose:", poses[count])
            else:
                color_image = color_frames[count].astype(np.uint8)
                depth_image = depth_frames[count]
                #depth_image = depth_frames[count]
                #color_image = color_frames[count].astype(np.uint8)
        
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
    parser.add_argument('-d','--duration', dest='duration',
                        type=int, default=10, help='Recording duration')
    parser.add_argument('-fps, --framerate', dest='framerate',
                        type=int, default=30, choices=[0,6,30,90], help='Recording framerate')
    parser.add_argument('-f', '--file', dest='file', required=True, help="Filename")


    args = parser.parse_args()
    main(args)
