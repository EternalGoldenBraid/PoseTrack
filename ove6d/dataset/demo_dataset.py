import os
import json
from pathlib import Path
from numba import njit
import cv2
import numpy as np
from scipy.spatial import Delaunay
import torch
from pytorch3d.io import load_ply
import open3d as o3d
from numpy.random import default_rng
from ipdb import iex

from ove6d.lib import rendering
#from lib.Sim3DR import RenderPipeline
#from lib.Sim3DR.Sim3DR import rasterize, get_normal

class Dataset():
    def __init__(self, data_dir, cfg, 
            cam_K, cam_height, cam_width, n_triangles, object_renderer=None):
        self.model_dir = Path(data_dir) / 'models_eval'
        self.cam_file = Path(data_dir) / 'camera.json'
        self.cam_K = cam_K
        self.cam_K_np = cam_K.cpu().numpy()
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.model_info = None
        self.obj_model_file = dict()
        self.obj_diameter = dict()
        self.point_cloud = dict()
        self.faces = dict()
        self.object_renderer = object_renderer
        self.cfg = cfg
        self.n_triangles: int = n_triangles # Number of points

        self.model_info_file = self.model_dir / 'models_info.json'
        #self.model_info_file = os.path.abspath(os.path.join(
            #os.path.dirname(__file__), '..', self.model_dir, 'models_info.json'))
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
        
        rng = default_rng()

        ### TODO: Combine faces and verts into same container.
        for model_file in sorted(self.model_dir.iterdir()):
            #breakpoint()
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']

                mesh = o3d.io.read_triangle_mesh(filename=str(model_file))
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangles)
                self.point_cloud[obj_id] = np.asanyarray(mesh.vertices)
                self.faces[obj_id] = np.asanyarray(mesh.triangles)

        if self.cam_K == None:
            print("Warning camera intrinsics not set.")

    @iex
    #@njit(parallel=True)
    def render_cloud(self, obj_id, R, t, image):

        ### FLIP Y, and Z coords
        #R = R@np.array([
        #                [1, 0, 0],
        #                [0, -1, 0],
        #                [0, 0, -1]], dtype=np.float32)
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        #P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t.T)
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t)

        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            return image, False

        #view_depth *= view_cam_info['depth_scale']
        #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
        #view_depth/=view_depth.max()
        
        P = P.astype(int)

        image[P[1], P[0], :] = 255
        return image, True

    def render_mesh(self, obj_id, R, t, image, color=(0,0,255)):
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t).squeeze()
        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            return image, False
        P = P.astype(np.int32)

        # Does not fill the triangels since all are passed as a single batch.
        # Iterating over earch triangle and rendering one by one results in fill.
        image = cv2.fillPoly(image, pts=np.array(P.T[self.faces[obj_id]][:,:,:2]), color=color)
        #image = cv2.drawContours(image=image, contourIdx=-1, color=color, thickness=-1, 
                #contours=np.array(P.T[self.faces[obj_id]][:,:,:2]))

        return image, True

    def render_depth(self, obj_id, R, t, mesh):
        self.obj_renderer = rendering.Renderer(width=self.cfg.RENDER_WIDTH, height=self.cfg.RENDER_HEIGHT)
        obj_context = rendering.SceneContext(obj=mesh, intrinsic=self.cam_K.cpu()) # define a scene
        obj_context.set_pose(rotation=R, translation=t)
        
        est_depth, est_mask = self.obj_renderer.render(obj_context)[1:]

        return est_depth
