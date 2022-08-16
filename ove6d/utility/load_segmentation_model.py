import os
import sys
import json
from pathlib import Path
import cv2
import torch
import numpy as np

from ipdb import iex
#from detectron2 import model_zoo
from detectron2.projects import point_rend

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from ove6d.lib import detectron_segmentation, contour_segmentation

def load(model, cfg, device, model_path=None):
    """
    TODO: ADD TYPES
    Segmentator needs to return:
        TODO

    ## Returns
    mask_gpu: torch.tensor on gpu
    mask: numpy.array
    scores: confidence scores. None if not provided.
    """
    if model == 'bgs_hsv':
        def segmentator_(image): 
            #im_hsv = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)
            im_hsv = cv2.cvtColor(
                    1./255*image.astype(np.float32), code=cv2.COLOR_BGR2HSV_FULL)
            sub = im_hsv.copy()
            low = 140; up = 220
            mask = cv2.inRange(sub, np.array([low,0,0]), np.array([up,1,1]))

            center = (cfg.RENDER_WIDTH//2, cfg.RENDER_HEIGHT//2)
            radius = cfg.RENDER_WIDTH // 4
            color = 255
            thickness = -1
            line_type = 8
            #circle = cv2.circle(mask, center, radius, color, thickness, line_type)
            circle = cv2.circle(np.zeros_like(mask, dtype=np.uint8), center, radius, color, thickness, line_type)//255
            mask = mask//255; mask=mask.astype(bool)
            mask = ~mask
            mask = circle*mask

            return mask.astype(np.uint8), torch.tensor(mask, device=device), None

        segmentator = segmentator_
    elif model == 'bgsKNN':
        def segmentator_(image):
            mask = cv2.createBackgroundSubtractorKNN().apply(image)
            return mask, torch.tensor(mask, device=device), None
        segmentator = segmentator_
    elif model == 'bgsMOG2':
        def segmentator_(image):
            mask = cv2.createBackgroundSubtractorMOG2().apply(image)
            return mask, torch.tensor(mask, device=device), None
        segmentator = segmentator_
    elif model == 'contour':
        none_array = np.array(())
        model = contour_segmentation.BackgroundContour()
        def segmentator_(image):
            #mask = contour_segmentation.ContourSegmentator().get_mask(image)//255
            masks = model.get_mask(image)
            if len(masks) == 0:
                return none_array, None, None
            return masks, torch.tensor(masks, device=device), None
        segmentator = segmentator_
    elif model == 'maskrcnn':

        none_array = np.array(())
        model_seg, model_cfg = detectron_segmentation.load_model_image_agnostic(
                model_path=model_path,
                #base_path+'/checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth',
                device=device, confidence=cfg.SEGMENTATION_CONFIDENCE)

        import detectron2.data.transforms as T
        aug = T.ResizeShortestEdge(
            [model_cfg.INPUT.MIN_SIZE_TEST, model_cfg.INPUT.MIN_SIZE_TEST], model_cfg.INPUT.MAX_SIZE_TEST
        )

        def segmentator_(image, model=model_seg, aug=aug):
            image = aug.get_transform(image).apply_image(image)
            inputs = {"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
                        "height": cfg.RENDER_HEIGHT, "width": cfg.RENDER_WIDTH}
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                pred = model([inputs])[0]['instances']
                mask_gpu = pred.get('pred_masks')
                scores = pred.get('scores')
                if mask_gpu.numel() == 0:
                    return none_array, None, None
                mask_cpu = mask_gpu.to(
                        non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
                return mask_cpu, mask_gpu, scores
        segmentator = segmentator_
    elif model == 'point_rend':
        none_array = np.array(())
        print("BASE:",base_path)
        #model_file = base_path+'/checkpoints/model_final_ba17b9_pointrend.pkl'
        model_path = base_path+'/checkpoints/model_final_edd263_pointrend.pkl'
        
        model_seg = detectron_segmentation.load_model_point_rend(model_path=model_path,
                config_yaml=base_path+ \
                        '/configs/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                #config_yaml='configs/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                confidence=0.7, base_path=base_path, device=device)
        def segmentator_(image, model=model_seg):
            pred = model(image)['instances']
            if pred.pred_masks.numel() == 0:
                return none_array, None, None
            #import pdb; pdb.set_trace()
            mask_gpu = pred.get('pred_masks')
            scores = pred.get('scores')
            mask_cpu = mask_gpu.to(
                    non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
            return mask_cpu, mask_gpu, scores
        segmentator = segmentator_
    else: 
        print("Invalid segmentation option:", model)
        return None

    return segmentator

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')
    parser.add_argument('-o', '--obj_id', dest='obj_id',
                        type=int, required=False,default=1,
                        help='Object index: {box, basket, headphones}')
    parser.add_argument('-b', '--buffer_size', dest='buffer_size',  
                        type=int, required=False, default=3,
                        help='Frame buffer for smoothing.')
    parser.add_argument('-n', '--n_points', dest='n_points',
                        type=int, required=False, default=2000,
                        help='Number of points for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='maskrcnn',
                        choices = ['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn',],
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
