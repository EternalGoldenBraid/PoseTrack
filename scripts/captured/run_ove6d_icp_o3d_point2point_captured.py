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
    file = "rs_records.hdf5"
    f = h5py.File(f"data/{file}", "r")
    if args.obj_id.name not in f[grp]:
        print("Object not found in recordings.")
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


    is_tracked = False
    has_init_pose = False
    alpha = 0.5 # For blending segmentated image.
    render_color = [(255, 0, 0), (0,0, 255)]
    H_init = torch.eye(4)
    threshold = 0.02
    new_cloud = o3d.geometry.PointCloud()
    old_cloud = o3d.geometry.PointCloud()
    icp_max_iters = 50

    for obj_recording in f[grp]:

        color_frames = f[grp][obj_recording]['color_frames'][:]
        depth_frames = f[grp][obj_recording]['depth_frames'][:]

        n_frames = color_frames.shape[0]
        duration = int(n_frames/fps)
        
        # Streaming loop
        for count in range(n_frames):
        
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
                    #if len(old_cloud.points) == 0:
                    if not is_tracked:
                        old_cloud.points = o3d.utility.Vector3dVector(pplane_ICP.depth_to_pointcloud(obj_depth, cam_K))
                    is_tracked = True

                    old_cloud = old_cloud.uniform_down_sample(every_k_points=4)
                    new_cloud = new_cloud.uniform_down_sample(every_k_points=4)

                    H_old = to_homo(R=R, t=t)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        old_cloud, new_cloud,
                        threshold, H_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iters)
                        )
                    dH = torch.tensor(reg_p2p.transformation).to(torch.float32)
                    print(dH)

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

                mask_ = masks.sum(axis=0, dtype=np.uint8)
                images = np.hstack([ 
                    color_image, 
                    cv2.addWeighted(depth_colormap, alpha, 255*mask_[...,None].repeat(repeats=3, axis=2), 1-alpha, 0.0)
                    ])
            else:
                images = np.hstack((color_image, depth_colormap))

            
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            #key = cv2.waitKey(int(1000/fps))
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('t') and not np.isnan(R).all():
                if not has_init_pose:
                    has_init_pose = True
                else: 
                    has_init_pose = False
                    is_tracked = False

    f.close()

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
                        choices=ObjectIds,
                        help='Object names')
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
    ### Python < 3.9 TODO: Set this up.
    #parser.add_argument('--feature', action='store_true', dest='render_mesh')
    #parser.add_argument('--no-feature', dest='render_mesh', action='store_false')
    #parser.set_defaults(render_mesh=True)
    ### Python >= 3.9
    parser.add_argument('-rm', '--render-mesh', dest='render_mesh', action=argparse.BooleanOptionalAction)
    parser.add_argument('-icp', dest='icp', action=argparse.BooleanOptionalAction)

    
    args = parser.parse_args()

    main(args)
