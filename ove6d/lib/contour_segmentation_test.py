import cv2
import numpy as np

# Parameters
blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
mask_color = (0.0,0.0,0.0)
mask_dilate_iter = 5
mask_erode_iter = 5

# initialize video from the webcam
video = cv2.VideoCapture(0)


### PRECOMPUTE??
# Set up mask with a matrix of 0's
mask = np.zeros([480,640], dtype = np.uint8)

# Get the area of the image as a comparison
image_area = mask.shape[0] * mask.shape[1]  

### MOVE THIS OUT
# calculate max and min areas in terms of pixels
max_area = max_area * image_area
min_area = min_area * image_area


while True:
    ret, frame = video.read()
    mask = np.zeros([480,640], dtype = np.uint8)
    if ret == True:
        # Convert image to grayscale        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, canny_low, canny_high)

        # Post process
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        # get the contours and their areas
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]


        ### MAP HERE?
        # Go through and find relevant contours and apply to mask
        for contour in contour_info:
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour[1] > min_area and contour[1] < max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour[0], (255))
                #import pdb; pdb.set_trace()


        ### POST PROCESSING USE THE ONELINER?
        # use dilate, erode, and blur to smooth out the mask
        mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
        mask = cv2.erode(mask, None, iterations=mask_erode_iter)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        # Ensures data types match up
        mask = (mask.astype('float32') / 255.0)
        frame = frame.astype('float32') / 255.0

        # Blend the image and the mask
        #masked = (mask[...,None] * frame) + ((1-mask[...,None]) * mask_color)
        #masked = (masked * 255).astype('uint8')
        #cv2.imshow("Foreground", np.hstack[(masked, (frame*255).astype('uint8'))])
        #import pdb; pdb.set_trace()
        #images = np.hstack((masked, (frame*255).astype('uint8')))

        edges = edges.astype('float32')/255
        edges = np.dstack( (edges, edges, edges) )
        #import pdb; pdb.set_trace()
        images = np.hstack((frame*mask[...,None], frame, edges))
        #cv2.imshow("Foreground", np.hstack([(masked, (frame*255).astype('uint8'))]))
        cv2.imshow("Foreground", images)
        #cv2.imshow("Foreground", mask)

        #import pdb; pdb.set_trace()

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
