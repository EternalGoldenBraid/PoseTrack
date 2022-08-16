import os
from time import time
import cv2
import numpy as np
from numba import prange, njit
import torch

class ContourSegmentator():
    def __init__(self):
        self.image_area = 480*640
        self.blur: int = 21
        self.canny_low: int = 15; self.canny_high: int = 150
        self.min_area: float = 0.0005; self.max_area: int = 0.95
        self.dilate_iter: int = 10; self.erode_iter: int = 10
        self.mask_color = (0.0,0.0,0.0);
        self.mask_dilate_iter: int = 5; self.mask_erode_iter: int = 5
        self.max_area = self.max_area * self.image_area
        self.min_area = self.min_area * self.image_area

    def get_mask(self, frame):
        # Convert image to grayscale        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, self.canny_low, self.canny_high)
    
        # Post process
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
    
        # get the contours and their areas
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    
        mask = np.zeros([480,640], dtype = np.uint8)
        ### MAP HERE?
        ## Go through and find relevant contours and apply to mask
        #for contour in contour_info:
        #    # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        #    if contour[1] > min_area and contour[1] < max_area:
        #        # Add contour to mask
        #        mask = cv2.fillConvexPoly(mask, contour[0], (255))
    
        ### MAP HERE?
        # Go through and find relevant contours and apply to mask
        #for contour in contour_info:
        for idx in prange(len(contour_info)):
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour_info[idx][1] > self.min_area and contour_info[idx][1] < self.max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour_info[idx][0], (255))
    
    
        ### POST PROCESSING USE THE ONELINER?
        # use dilate, erode, and blur to smooth out the mask
        mask = cv2.dilate(mask, None, iterations=self.mask_dilate_iter)
        mask = cv2.erode(mask, None, iterations=self.mask_erode_iter)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
    
        return mask

#from torchvision import datasets, models, transforms
#import torch.nn as nn
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

class BackgroundContour():

    def __init__(self, model_filepath=None):
        self.model_filepath = model_filepath
        self.detection_type = "contours"
        self.trained = False
        self.depth_raw = None
        self.lr = -1
        self.bridge = None
        self.initFlag = False
        #self.transforms = A.Compose(
        #    [
        #        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #        A.Resize(224, 224),
        #        ToTensorV2(),
        #    ])
        #self.model = models.resnet50(pretrained=True)
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, 10)
        #self.model = self.model.to('cpu')
        #self.model.eval()
        #self.model.load_state_dict(torch.load(self.model_filepath, map_location=torch.device('cpu')))
        self.classnames = ['garbage','gear','gear_side','bottom_casing',
                           'bottom_casing_side','bottom_casing_inv','top_casing',
                           'top_casing_inv','two_gears','two_gears_on_bottom_casing']


    def get_mask(self, image):
        #result = []
        fgMask = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        blur = cv2.blur(image,(9,9))

        if not self.initFlag:
            self.bg = blur
            self.initFlag = True
        if self.initFlag:
            t1 = np.float32(np.mean(self.bg,-1))
            t2 = np.float32(np.mean(blur,-1))
            tmp = np.abs(t1-t2)
            tmp = tmp>30
            fgMask = np.uint8(tmp)*255

        #contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = np.array([cv2.contourArea(cnt) for cnt in contours])
        contours = [contours[i] for i in range(len(contours)) if area[i] > 100]

        #masks = np.array([cv2.drawContours(np.zeros((480, 640)), c, -1, (1,1,1), 3) for c in contours])
        masks = np.array([cv2.drawContours(np.zeros((480, 640)), [c], -1, (1,1,1), cv2.FILLED) for c in contours])
        #masks = cv2.drawContours(np.zeros((480, 640)), contours, -1, (1,1,1), 3)[...,None]
        return masks.astype(int).astype(np.uint8)
        #return (masks).astype(np.uint8)

        #for contour in contours:
        #    x,y,w,h = cv2.boundingRect(contour)
        #    im = image[y:y+h,x:x+w,:]
        #    im_torch = self.transforms(image=im)
        #    im_torch = im_torch['image'][None,...]
        #    with torch.no_grad():
        #        out = self.model(im_torch)
        #        _, preds = torch.max(out, 1)
        #        probs = torch.softmax(out,1)
        #    objectclass = self.classnames[preds]
        #    if preds != 0:
        #        result.append((objectclass, contour))
        #return result


if __name__=='__main__':

    import sys
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
    base_path = os.path.dirname(os.path.abspath("."))
    print("Base_path:", base_path)
    sys.path.append(base_path)

    from utility.cam_control import Camera

    if True:
        ### REALSENSE
        cam = Camera(size=(640, 480), framerate=60)
        depth_scale, cam_K =  cam.depth_scale, cam.cam_K

        segs = [ContourSegmentator(), BackgroundContour()]
        seg = segs[1]
        while True:
            depth_image, color_image = cam.get_image()
            depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            masks = seg.get_mask(color_image)
            if len(masks) != 0:
                images = np.concatenate((color_image, color_image*masks.sum(axis=0, dtype=np.uint8)[...,None]), axis=1)
                #images = np.concatenate((color_image, depth_colormap*masks.sum(axis=0, dtype=np.uint8)[...,None]), axis=1)
                cv2.imshow("Foreground", images)
            else: 
                images = np.concatenate((color_image, depth_colormap), axis=1)

            cv2.imshow("Foreground", images)

            ## Use the q button to quit the operation
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        del cam
    else:
        # initialize video from the webcam
        #cam = cv2.VideoCapture(cv2.CAP_V4L2)
        cam = cv2.VideoCapture(0)
        segs = [ContourSegmentator(), BackgroundContour()]
        seg = segs[1]
        while True:
            start = time()
            #ret, frame = cam.read()
            depth_image, color_image = cam.get_image()
            if ret == True:
        
                masks = seg.get_mask(frame)

                if len(masks) == 0:
                    continue

                masks = masks[0]
        
                ## Ensures data types match up
                #mask = (mask.astype('float32') / 255.0)
                #frame = frame.astype('float32') / 255.0
        
                ## Blend the image and the mask
                ##masked = (mask[...,None] * frame) + ((1-mask[...,None]) * mask_color)
                ##masked = (masked * 255).astype('uint8')
                ##cv2.imshow("Foreground", np.hstack[(masked, (frame*255).astype('uint8'))])
                ##import pdb; pdb.set_trace()
                ##images = np.hstack((masked, (frame*255).astype('uint8')))
        
                #edges = edges.astype('float32')/255
                #edges = np.dstack( (edges, edges, edges) )

                images = np.hstack((frame*masks[...,None], frame))
                #cv2.imshow("Foreground", np.hstack([(masked, (frame*255).astype('uint8'))]))
                cv2.imshow("Foreground", images)
                #cv2.imshow("Foreground", mask)
        
                print("Time: ", time()-start)
        
                ## Use the q button to quit the operation
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
            else: 
                continue
    
        cv2.destroyAllWindows()
        video.release()
