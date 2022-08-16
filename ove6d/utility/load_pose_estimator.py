from ove6d.lib import ove6d, rendering, network
import torch

class PoseEstimator:
    """
    Represents OVE6D Pose estimation model.
    Loads object codebooks and offers an api for requeting poses based on images.
    TODO : WHAT IMAGES?
    """
    def __init__(self, cfg, cam_K, dataset, codebook_path,  model_path, obj_id=None, obj_ids = [], device='cpu'):
        self.obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)
        self.cam_K = cam_K
        self.device = device
        self.cfg = cfg
        self.dataset = dataset
        self.obj_id = obj_id
        self.obj_codebook = None 
        self.model_net = None
        self.obj_context = None

        # Load model
        #ckpt_file = pjoin(base_path, 'checkpoints', "OVE6D_pose_model.pth")
        self.model_net = network.OVE6D().to(self.device)
        self.model_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_net.eval()
        print('OVE6D has been loaded!')

        # Load object codebooks for all objects
        self.object_codebooks = ove6d.OVE6D_codebook_generation(codebook_dir=codebook_path,
                                                            model_func=self.model_net,
                                                            dataset=self.dataset,
                                                            config=self.cfg,
                                                            device=self.device)

        print("Object codebooks have been loaded with id's:")
        print(self.object_codebooks.keys())

        if len(obj_ids) == 0 and not obj_id:
            print("Not enough object ids'")
            raise RuntimeError
        elif len(obj_ids) == 1 or obj_id is not None:
            if len(obj_ids) == 1:
                self.obj_id = obj_ids[0]
            elif obj_id is not None:
                self.obj_id = obj_id
            self.obj_codebook = self.object_codebooks[self.obj_id]

            # TODO: Which mesh model to use? From codebook or from *.ply?
            #obj_context = rendering.scenecontext(obj=obj_mesh, intrinsic=cam_k.cpu()) # define a scene
            self.obj_context = rendering.SceneContext(obj=self.obj_codebook['obj_mesh'], intrinsic=cam_K.cpu()) # define a scene
        else:
            print("Multiobject support not yet implemented")
            raise NotImplementedError

    
    def estimate_pose(self, obj_depth, obj_mask):
        """
        TODO Replace this with @static_object
        Estimate pose for predefined object.
        B*height*width
        """

        #tar_obj_depth = (view_depth * obj_mask).squeeze()
        pose_ret = ove6d.OVE6D_mask_full_pose(
            model_func=self.model_net, 
            #obj_depth=tar_obj_depth[None,:,:],
            obj_depth=obj_depth, # 1*h*w
            obj_mask=obj_mask, # 1*h*w
            obj_codebook=self.obj_codebook,
            cam_K=self.cam_K,
            config=self.cfg,
            obj_renderer=self.obj_renderer,
            device=self.device,
            obj_context=self.obj_context)

        return pose_ret['raw_R'][None,...], pose_ret['raw_t'][None,...]

    def estimate_poses(self, obj_depths, obj_masks, scores):

        #tar_obj_depth = (view_depth * obj_mask).squeeze()
        pose_ret = ove6d.OVE6D_rcnn_full_pose(
            model_func=self.model_net, 
            #obj_depth=tar_obj_depth[None,:,:],
            obj_depths=obj_depths, # N*h*w
            obj_masks=obj_masks, # N*h*w
            obj_rcnn_scores=scores,
            obj_codebook=self.obj_codebook,
            cam_K=self.cam_K,
            config=self.cfg,
            obj_renderer=self.obj_renderer,
            device=self.device,
            return_rcnn_idx=False # TODO role?
            )

        # TODO Add multiobject support.
        return pose_ret['raw_R'][None,...], pose_ret['raw_t'][None,...]

    def __del__(self):
        del self.obj_renderer
