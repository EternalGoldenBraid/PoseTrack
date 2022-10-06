import os
from os.path import join as pjoin
from pathlib import Path
from time import perf_counter
import sys
import warnings
import json

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)



#breakpoint()

import cv2
import numpy as np
import h5py
from ipdb import iex
from tqdm import tqdm
import mediapy as media

breakpoint()
from ove6d.configs import config as cfg

from io_utils import annotate_images
titles = [
        ["OVE6D initialized, ICP track, full cad model", "OVE6D track, full cad model"],
        ["OVE6D initialzide, ICP track, simpler model", "OVE6D track, simpler cad model"]
        ]


def main(args):

    grp = args.group
    file = f"{args.file}.hdf5"

    f = h5py.File(f"data/renderings/{file}", "r")
    depth_scale = f['meta']["depth_scale"][:]
    cam_K = f['meta']["camera_intrinsic"][:]

    fps = f['meta']["framerate"][0]
    #if args.framerate == 0:
    #    fps = f['meta']["framerate"][0]
    #else:
    #    fps = args.framerate

    print("Playback data:", file, grp, " at framerate:", fps)

    for object_recording in f[grp]:
        print("Recorded object:", object_recording)

        # Load object frames
        obj_path = Path(f.name, grp, object_recording)
        rendered_frames = False
        rm = ['color_frames', 'depth_frames']
        cad_list = [i for i in f[str(obj_path)].keys() if i not in rm]

        #if 'rendered_frames' in f[str(obj_path)].keys():
        if len(cad_list) > 0:
            # Rendered frames exist for some cad model(s).
            rendered_frames = True
            
            # Combine rendered cad models to single window.

            # Stack cad models vertically (axis=1) and trackers horizontally (axis=2).
            for idx, cad_model in enumerate(cad_list):
                trackers = f[str(obj_path/cad_model)].keys()
                if idx == 0:
                    trackers_frames = np.concatenate(
                            np.array([f[str(obj_path/cad_model/tracker/'rendered_frames')] for tracker in trackers]) ,axis=2)
                else:
                    next_tracker_frames = np.concatenate(
                            np.array([f[str(obj_path/cad_model/tracker/'rendered_frames')] for tracker in trackers]) ,axis=2)
                    frames = np.concatenate(( trackers_frames, next_tracker_frames), axis=1)

                #frames = f[str(obj_path)]['rendered_frames'][:][None, ...]
                #poses = f[str(obj_path/'poses')][:]

            # If only one cad model there's only one horizontal stack.
            if idx == 0: frames = trackers_frames

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

                # Add annotations
                frames[count] = annotate_images( images=frames[count], titles=titles,
                                            img_size=tuple(f['meta/depth_shape'][:])
                                         )
                images = frames[count].astype(np.uint8)
                #print("Pose:", poses[count])

            else:
                color_image = color_frames[count].astype(np.uint8)
                depth_image = depth_frames[count]
                #depth_image = depth_frames[count]
                #color_image = color_frames[count].astype(np.uint8)
        
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
        
                images = np.hstack((color_image, depth_colormap))

                # Add annotations
                images = annotate_images( images=images, titles=titles,
                                            img_size=tuple(f['meta/depth_shape'][:])
                                         )
            
            if args.show_window == True:
                #cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', images)
                key = cv2.waitKey(int(1000/fps))
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    break
                if key & 0xFF == ord('s'):
                    args.show_window = False

        cv2.destroyAllWindows()

        # Save as gif
        # TODO Add flag
        if True:
            #conv = lambda arr: cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            #frames = np.array(list(map(conv, frames))) # TODO Parallel?
            frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames.astype(np.uint8)])
            save_path: Path = Path("data", "renderings", args.file)

            # Save video
            media.write_video(save_path.with_suffix('.mp4'), frames, fps=fps)

            # Save gif
            #from PIL import Image
            #gif_path = f"data/renderings/{args.file}.gif"
            #print("Saving gif to:", gif_path)
            #imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode='RGB') for frame in frames.astype(np.uint8)]
            ## duration is the number of milliseconds between frames; this is 40 frames per second
            #imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)

            print("Done")
        
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
    #parser.add_argument('-fps, --framerate', dest='framerate',
    #                    type=int, default=30, choices=[0,6,30,90], help='Recording framerate')
    parser.add_argument('-f', '--file', dest='file', required=True, help="Filename")
    parser.add_argument('--show_window', action=argparse.BooleanOptionalAction, default=True)


    args = parser.parse_args()
    main(args)
