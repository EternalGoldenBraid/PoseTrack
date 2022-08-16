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

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from ove6d.utility import load_segmentation_model, cam_control

from ove6d.configs import config as cfg

def main(args):

    # Load camera module
    #timeit.log("Realsense initialization.")
    # TODO change to row_major for numpy...? What's torch
    fps = args.framerate
    duration = args.duration
    n_frames = int(fps*duration)
    color_frames = np.zeros((n_frames, cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH, 3), dtype=np.uint8)
    depth_frames = np.zeros((n_frames, cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH), dtype=np.float32)

    cam = cam_control.Camera(size=(cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT), framerate=fps)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K
    cam_K_np = cam_K.numpy()
    recording = False

    # load segmentation module
    segmentator = load_segmentation_model.load(
        model=args.segment_method, cfg=cfg, device='cuda',
        model_path='ove6d/checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth'
        #model_path=str(Path('ove6d','checkpoints','FAT_trained_Ml2R_bin_fine_tuned.pth'))
        )

    # Streaming loop
    alpha = 0.5
    try:
        count = -1
        while True:
            count += 1

            fps_start = perf_counter()
            
            depth_image, color_image = cam.get_image()

            if recording:
                masks = np.array([])
            else: 
                masks, masks_gpu, scores = segmentator(color_image)

            if recording:
                color_frames[count] = color_image
                depth_frames[count] = depth_image

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 

            if masks.size != 0:
                mask_ = masks.sum(axis=0, dtype=np.uint8)
                images = np.hstack([
                            color_image,
                            cv2.addWeighted(depth_colormap, alpha, (255*mask_)[...,None].repeat(repeats=3, axis=2), 1-alpha, 0.0)
                            ])
            else:
                images = np.hstack((color_image, depth_colormap))

            
            cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
            if key & 0xFF == ord('r'):
                print("Recording")
                recording = True
                count = -1
            if recording and count == n_frames-1:
                break

        cv2.destroyAllWindows()

        if not recording:
            print("No recorded frames")
            return

        filename = "_".join([obj.name for obj in (args.obj_id)])
        
        with h5py.File(f"data/{args.file}.hdf5", "a") as f:
            if len(args.obj_id) == 1:
                group = "single_object"
            else:
                group = "multi_object"

            if args.occlusion:
                group += "_occluded"

            if not group in f:
                print("Creating group:", group)
                f.create_group(group)

            # Store metadata
            if not 'meta' in f:
                f.create_group('meta')
                f['meta'].create_dataset(name="depth_scale", shape=1, dtype='f')
                f['meta'].create_dataset(name="camera_intrinsic", shape=cam_K.numpy().shape, dtype='f')
                f['meta'].create_dataset(name="framerate", shape=1, dtype='i')

            print("Saving to scenario:", group)
            if filename not in f[group]:
                print("Creating dataset:", filename)
                #f[group].create_dataset(name=filename+"/color_frames", shape=color_frames.shape, dtype='i8')
                #f[group].create_dataset(name=filename+"/depth_frames", shape=depth_frames.shape, dtype='f')
                path_ = Path(group,filename)
                f.create_dataset(name=str(path_/"color_frames"), shape=color_frames.shape, dtype='i8')
                f.create_dataset(name=str(path_/"depth_frames"), shape=depth_frames.shape, dtype='f')

            #f[group][filename+"/color_frames"][:] = color_frames
            #f[group][filename+"/depth_frames"][:] = depth_frames
            f[str(path_/"color_frames")][:] = color_frames
            f[str(path_/"depth_frames")][:] = depth_frames
            f['meta']["depth_scale"][:] = depth_scale
            f['meta']["camera_intrinsic"][:] = cam_K.numpy()
            f['meta']["framerate"][:] = fps

        print(f"Saved {n_frames} frames of shape {depth_frames.shape} and {color_frames.shape} to {filename}.")

    finally:
        del cam

if __name__=="__main__":
    import argparse

    from enum import Enum, unique
    class ArgTypeMixin(Enum):

        @classmethod
        def argtype(cls, s: str) -> Enum:
            try:
                return cls[s]
            except KeyError:
                raise argparse.ArgumentTypeError(
                    f"{s!r} is not a valid {cls.__name__}")

        def __str__(self):
            return self.name

    @unique
    class ObjectIds(ArgTypeMixin, Enum):
        box = 1
        head_phones = 3
        engine_main = 4
        dual_sphere = 5
        tea = 6
        bolt = 7
        wrench = 8
        lego = 9
        eraser_lowq = 10
        eraser_highq = 11
        #eraser_lowq = 10

    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')

    parser.add_argument('-o','--object', dest='obj_id',
                        type=ObjectIds.argtype, default=ObjectIds.box, 
                        nargs='+', choices=ObjectIds,
                        help='Object names')
    parser.add_argument('-d','--duration', dest='duration',
                        type=int, default=10, help='Recording duration in seconds.')
    parser.add_argument('--fps, --framerate', dest='framerate',
                        type=int, default=60, choices=[6,30,60], help='Recording framerate.')
    parser.add_argument('-f', '--file', dest='file', type=str, required=True, help="Filename")

    ### Python < 3.9 TODO: Set this up.
    #parser.add_argument('--feature', action='store_true', dest='render_mesh')
    #parser.add_argument('--no-feature', dest='render_mesh', action='store_false')
    #parser.set_defaults(render_mesh=True)
    ### Python >= 3.9

    parser.add_argument('--occlusion', dest='occlusion', action=argparse.BooleanOptionalAction)
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                    required=False, default='maskrcnn',
                    choices = ['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
                    help="""Method of segmentation.
                    contour: OpenCV based edge detection ...,
                    TODO:
                    """)


    
    args = parser.parse_args()

    main(args)
