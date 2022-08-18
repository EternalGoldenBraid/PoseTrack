import os
from time import perf_counter
import sys
import warnings
import json
from itertools import product
from pathlib import Path
from numba import njit, prange
import h5py
import cv2
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
import pyrealsense2 as rs
from scipy import stats
from matplotlib import pyplot as plt
import open3d as o3d
from tqdm import tqdm


from ipdb import iex

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)


from ove6d.utility import timeit, load_segmentation_model, cam_control
from ove6d.utility.load_pose_estimator import PoseEstimator

from ove6d.lib.render_cloud import load_cloud, render_cloud # TODO Use this
from ove6d.lib import ove6d, pplane_ICP

from ove6d.dataset import demo_dataset
from ove6d.configs import config as cfg

from tracknet.utils import load_tracker

#DEVICE = torch.device('cuda')
#DEVICE = torch.device('cpu')
#DEVICE = 'cpu'
DEVICE = 'cuda'


def main(args):

    cfg.DATASET_NAME = 'huawei_box'        # dataset name
    cfg.USE_ICP = args.icp

    # Load data
    grp = args.group
    file_in = f"{args.file_in}.hdf5"
    f = h5py.File(f"data/recordings/{file_in}", "r")
    if args.obj_name not in f[grp]:
        print("Object name not found in recordings:", f[grp].keys())
        return -1
    depth_scale = f['meta']["depth_scale"][:]
    cam_K = torch.tensor(f['meta']["camera_intrinsic"][:])
    fps = f['meta']["framerate"][0]

    # load segmentation module
    segmentator = load_segmentation_model.load(
        model=args.segment_method, cfg=cfg, device=DEVICE,
        model_path='ove6d/checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth'
        #model_path=str(Path('ove6d','checkpoints','FAT_trained_Ml2R_bin_fine_tuned.pth'))
        )

    # Load mesh models
    #dataroot = Path(os.path.dirname(__file__)).parent/Path(cfg.DATA_PATH)
    #dataroot = Path(os.path.realpath(__file__)).parent.parent/Path(cfg.DATA_PATH)
    dataroot = Path('ove6d','Dataspace')
    # TODO: Change this torch tensor to numpy array as it comes from the camera.
    dataset = demo_dataset.Dataset(data_dir=dataroot/ 'huawei_box', cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH,
                n_triangles=args.n_triangles)

    # Load pose estimation module
    obj_id: int = args.obj_id.value
    #import pdb; pdb.set_trace()
    codebook_path = pjoin(base_path,'Dataspace/object_codebooks',
        cfg.DATASET_NAME,
        'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR),
        'views_{}'.format(str(cfg.RENDER_NUM_VIEWS)))

    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, obj_id=obj_id,
            model_path=Path('ove6d','checkpoints','OVE6D_pose_model.pth'),
            device=DEVICE, dataset=dataset, codebook_path=codebook_path)

    # TODO
    # Passing initialized renderer? Implications?
    dataset.object_renderer = pose_estimator.obj_renderer

    # Streaming loop

    mod_count: int = 0
    buffer_size: int = args.buffer_size
    frame_buffer = np.empty([buffer_size, cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH])
    R = np.ones((3,3))*np.nan
    T = torch.eye(4)

    def to_homo(R,t,T=T):
        T[:3,:3]=R; T[0:3,3]=t;
        return T


    ## Tracking setup
    is_tracked = False
    has_init_pose = False
    H_init = torch.eye(4)
    threshold = 0.02
    new_cloud = o3d.geometry.PointCloud()
    old_cloud = o3d.geometry.PointCloud()
    icp_max_iters = 50

    ## Playback setup
    alpha = 0.5 # For blending segmentated image.
    render_color = [(255, 0, 0), (0,0, 255)]
    to_draw = True

    ## Load recordings
    #for args.obj_name in f[grp][args.obj_name]:
    obj_path = Path(f.name, grp, args.obj_name)
    color_frames = f[str(obj_path/'color_frames')][:]
    depth_frames = f[str(obj_path/'depth_frames')][:]
    f.close()
    rendered_frames = np.empty_like(color_frames, dtype=np.uint8)

    n_frames = color_frames.shape[0]
    duration = int(n_frames/fps)

    poses = np.eye(4, 4)[None,...].repeat(repeats=n_frames, axis=0) # In homo form.

    ## CV2 window Keybinds
    save = 'r'
    save_track = 's'
    track = 't'
    quit = 'q'
    print("Save pose estimation without tracking:", save)
    print("Save and track:", save_track)
    print("Track without saving", track)
    print("Quit", quit)

    # Streaming loop
    trackers = ['ove6d', 'icp']
    for tracker in trackers:
        for count in tqdm(range(n_frames)):
        
            depth_image = depth_frames[count]
            color_image = color_frames[count].astype(np.uint8)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
            masks, masks_gpu, scores = segmentator(color_image)

            if masks.size != 0:
                if has_init_pose:

                    current_frame = color_image.copy()
                    new_obj_depth=torch.tensor(
                        (depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()
                    new_cloud.points = o3d.utility.Vector3dVector(pplane_ICP.depth_to_pointcloud(new_obj_depth, cam_K))

                    # If previous pose is given by pose estimator, ICP will be done on next.
                    # On subsequent iterations old_cloud will be ICP transformed 
                    if not is_tracked:
                        old_cloud.points = o3d.utility.Vector3dVector(pplane_ICP.depth_to_pointcloud(obj_depth, cam_K))
                    is_tracked = True

                    #old_cloud = old_cloud.uniform_down_sample(every_k_points=4)
                    #new_cloud = new_cloud.uniform_down_sample(every_k_points=4)
                    old_cloud.estimate_normals() # Default knn k=30
                    new_cloud.estimate_normals() # Default knn k=30


                    H_old = to_homo(R=R, t=t)
                    poses[count] = H_old

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        old_cloud, new_cloud,
                        threshold, H_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iters)
                        )
                    dH = torch.tensor(reg_p2p.transformation).to(torch.float32)

                    H = (dH @ H_old)
                    R = H[:3,:3][None,...]; t = H[:3,3][None,None,...]
                    # Rotate and translate previous frame to match new frames pointcloud.
                    old_cloud.points = o3d.utility.Vector3dVector(
                        (dH[:3,:3].to(torch.float32) @ np.asarray(old_cloud.points).T + dH[:3,3][...,None].to(torch.float32)).T)

                    color = render_color[0]
                    #color_image, done = dataset.render_mesh(obj_id=obj_id, 
                    #         R=R[0].numpy().astype(np.float32), 
                    #         t=t[0].numpy()[...,None].astype(np.float32),
                    #         image=current_frame.copy(), color=color)

                    #images = np.hstack([ 
                    #    color_image, 
                    #    cv2.addWeighted(depth_colormap, alpha, 255*mask_[...,None].repeat(repeats=3, axis=2), 1-alpha, 0.0)
                    #    #color_image*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                    #    ])
                else:

                    ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
                    obj_depth=torch.tensor((depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()
                    R, t = pose_estimator.estimate_pose(obj_mask=masks_gpu[0][None,...],
                                obj_depth=obj_depth[None,...])

                    ### TODO Multi object support.
                    #obj_depths = torch.tensor([(depth_image*mask*depth_scale).astype(np.float32) for mask in masks])
                    #R, t = pose_estimator.estimate_poses(obj_masks=masks_gpu, scores=scores,
                    #            obj_depths=obj_depths.squeeze())

                    color = render_color[1]

                for transform_idx in range(R.shape[0]):
                    #color_image, done = dataset.render_cloud(obj_id=obj_id, 
                    #        R=R[transform_idx].numpy().astype(np.float32), 
                    #        t=t[transform_idx].numpy()[...,None].astype(np.float32),
                    #        image=color_image)

                    color_image, done = dataset.render_mesh(obj_id=obj_id, 
                             R=R[transform_idx].numpy().astype(np.float32), 
                             t=t[transform_idx].numpy()[...,None].astype(np.float32),
                             image=color_image.copy(), color=color)

                rendered_frames[count] = color_image
                mask_ = masks.sum(axis=0, dtype=np.uint8)
                images = np.hstack([ 
                    color_image, 
                    cv2.addWeighted(depth_colormap, alpha, 255*mask_[...,None].repeat(repeats=3, axis=2), 1-alpha, 0.0)
                    ])
            else:
                images = np.hstack((color_image, depth_colormap))

            if to_draw:
                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', images)
            #key = cv2.waitKey(int(1000/fps))
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord(quit) or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord(track) and not np.isnan(R).all():
                if not has_init_pose:
                    has_init_pose = True
                else: 
                    has_init_pose = False
                    is_tracked = False
            elif key & 0xFF == ord(save_track):
                # Skip intermediate drawing and process all frames in a batch.
                if not has_init_pose:
                    has_init_pose = True
                to_draw = False
                cv2.destroyAllWindows()
            elif key & 0xFF == ord(save):
                # Skip intermediate drawing and process all frames in a batch.
                to_draw = False
                cv2.destroyAllWindows()


        if not to_draw:
            #if args.save_rendered:
            if True:
                file_out = f"{args.file_out}.hdf5"
                f = h5py.File(f"data/renderings/{file_out}", "a")
                cad_path = Path(f"{grp}/{args.obj_name}/{args.obj_id.name}")

                if not grp in f:
                    print("Creating group:", grp)
                    f.create_group(grp)

                    with h5py.File(f"data/recordings/{file_in}", "r") as f_in:
                        #f.create_dataset('meta', data=f_in['meta'])
                        f_in.copy('meta', f)

                # TODO optimize this.
                if args.obj_name not in f[grp].keys():
                    for tracker_ in trackers:
                        f.create_dataset(name=str(cad_path/tracker_/'rendered_frames'), shape=rendered_frames.shape, dtype='i')
                        f.create_dataset(name=str(cad_path/tracker_/'poses'), shape=poses.shape, dtype='f')
                elif args.obj_id.name not in f[grp][args.obj_name]:
                    for tracker_ in trackers:
                        f.create_dataset(name=str(cad_path/tracker_/'rendered_frames'), shape=rendered_frames.shape, dtype='i')
                        f.create_dataset(name=str(cad_path/tracker_/'poses'), shape=poses.shape, dtype='f')

                f[f"{grp}/{args.obj_name}/{args.obj_id.name}/{tracker}/rendered_frames"][:]=rendered_frames
                f[f"{grp}/{args.obj_name}/{args.obj_id.name}/{tracker}/poses"][:]=poses

                print("Saved rendered frames to file", args.file_out+".hd5f", " in path:", 
                        obj_path/args.obj_id.name/tracker/'rendered_frames', "and",
                        obj_path/args.obj_id.name/tracker/'poses')

            for frame in rendered_frames:
                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', frame)
                key = cv2.waitKey(int(1000/fps))
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

            #r = input("Save as video? [y/n]")
            #if r.lower() == 'y':
            #    name = input("name: ")
            #    writer = cv2.VideoWriter(f"{name}.avi",
            #    #cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))
            #    cv2.VideoWriter_fourcc(*"MJPG"), fps,(rendered_frames.shape[1], rendered_frames.shape[0]))

            #    for frame in rendered_frames:
            #        #writer.write(np.random.randint(0, 255, (480,640,3)).astype('uint8'))
            #        writer.write(frame)

            #    writer.release()
        f.close()
        
        # This for next round icp if first run was ove6d only.
        to_draw = True




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
        box_synth = 12


    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')

    parser.add_argument('-o','--object', dest='obj_id',
                        type=ObjectIds.argtype, default=ObjectIds.box, choices=ObjectIds,
                        help='Object cad model to be used for pose estimation and rendering.')
    parser.add_argument('--object_name', dest='obj_name',
                        type=str, help='Recorded object name.')

    parser.add_argument('--scenario', dest='group',
                        type=str, default='single_object',
                        choices=['single_object', 'single_object_occluded',
                            'multi_object', 'multi_object_occluded'],
                        help='Scenarios')
    parser.add_argument('-b', '--buffer_size', dest='buffer_size',  
                        type=int, required=False, default=3,
                        help='Frame buffer for smoothing.')
    parser.add_argument('-n', '--n_triangles', dest='n_triangles',
                        type=int, required=False, default=2000,
                        help='Number of triangles for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='maskrcnn',
                        choices = ['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
                        help="""Method of segmentation.
                        contour: OpenCV based edge detection ...,
                        TODO:
                        """)
    parser.add_argument('-fin', '--file_in', dest='file_in', required=True, help="Filename of the recording.")
    parser.add_argument('-fout', '--file_out', dest='file_out', required=True, help="Filename of the output.")
    ### Python < 3.9 TODO: Set this up.
    #parser.add_argument('--feature', action='store_true', dest='render_mesh')
    #parser.add_argument('--no-feature', dest='render_mesh', action='store_false')
    #parser.set_defaults(render_mesh=True)
    ### Python >= 3.9
    parser.add_argument('-rm', '--render-mesh', dest='render_mesh', action=argparse.BooleanOptionalAction)
    parser.add_argument('-icp', dest='icp', action=argparse.BooleanOptionalAction)

    
    args = parser.parse_args()

    main(args)
