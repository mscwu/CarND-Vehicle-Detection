import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

v1 = (187, 720)
v2 = (591, 450)
v3 = (688, 450)
v4 = (1122, 720)
mask_vertices = np.array([[(115, 720),(605, 420), (690, 420), (1220, 720)]], dtype=np.int32)
lower_yellow = np.array([0, 70, 112])
upper_yellow = np.array([45, 228, 255]) 
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255]) 
sobel_x_threshold = [30, 50]
ang_threshold = [0.9, 1.1]
kernel_size = 7
quick_search_margin = 100
nwindows = 16 
sliding_margin = 70
minpix = 100


def import_camera_data():
    try:
        data = pickle.load(open('camera.pickle', 'rb'))
        print("Calibration exists. File loaded.")
        objpoints = data['objpoints']
        imgpoints = data['imgpoints']
        return objpoints, imgpoints
    except (OSError, IOError) as e:
        print("Cannot find camera data in current directory!")

def cal_undistort(img, objpoints, imgpoints):
    """Return undistorted image
    
    Keyword arguments:
    img -- image to be undistorted
    objpoints -- 3d points in real world space
    imgpoints -- 2d points in image plane
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def pick_yellow(img, lower_yellow, upper_yellow, return_binary=False):
    # Convert BGR to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)   

    # Threshold the HLS image to get only yellow colors
    mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)
    if return_binary:
        return mask
    else:
        return res

def pick_white(img, lower_white, upper_white, return_binary=False):
    # Threshold the BGR image to get only white colors
    mask = cv2.inRange(img, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)
    if return_binary:
        return mask
    else:
        return res

def pick_white_yellow(img, lower_yellow, upper_yellow, lower_white, upper_white, return_binary=False):
    white = pick_white(img, lower_white, upper_white, True)
    yellow = pick_yellow(img, lower_yellow, upper_yellow, True)
    color_mask = cv2.bitwise_or(white, yellow)
    res = cv2.bitwise_and(img, img, mask = color_mask)
    if return_binary:
        return color_mask
    else:
        return res

def sobel_x_gradient(img, k_threshold, return_binary=False):
    r_channel = img[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=9) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= k_threshold[0]) & (scaled_sobelx <= k_threshold[1])] = 1
    
    res = cv2.bitwise_and(img, img, mask = sxbinary)
    if return_binary:
        return sxbinary
    else:
        return res

def dir_threshold(img, sobel_kernel=3, thresh=[.7, 1.3]):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r_channel = img[:,:,2]
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

def combined_gradient(img, k_threshold=[20, 255], ang_threshold=[0.9, 1.2], kernel_size=15):
    sobelx = sobel_x_gradient(img, k_threshold, True)
    grad_dir = dir_threshold(img, kernel_size, ang_threshold)
    binary_output =  np.zeros_like(sobelx)
    binary_output[(sobelx == 1) & (grad_dir == 1)] = 1
    
    return binary_output

def binary_img(img, lower_yellow, upper_yellow, lower_white, upper_white, 
               k_threshold=[20, 255], ang_threshold=[0.9, 1.2], kernel_size=15):
    color_binary = pick_white_yellow(img, lower_yellow, upper_yellow, lower_white, upper_white, True)
    gradient_binary = combined_gradient(img, k_threshold, ang_threshold, kernel_size)
    binary_output =  np.zeros_like(color_binary)
    binary_output[(color_binary/255 == 1) |  (gradient_binary == 1)] = 1   
    return binary_output

def draw_region(img, v1, v2, v3, v4):
    img = np.copy(img)
    cv2.line(img, v1, v2, color=[0, 0, 255], thickness=2)
    cv2.line(img, v2, v3, color=[0, 0, 255], thickness=2)
    cv2.line(img, v3, v4, color=[0, 0, 255], thickness=2)
    cv2.line(img, v4, v1, color=[0, 0, 255], thickness=2)
    return img

def warp(img):
    '''Compute perspective transformation M and its inverse and a warped image
    
    Keyword arguments:
    img -- input image
    '''
    img = np.copy(img)
    img_size = (img.shape[1], img.shape[0])
    # source points
    src = np.float32([v1, v2, v3, v4])
    # desination points
    dst = np.float32([[450, img.shape[0]], [450, 0], [img.shape[1]-450, 0], [img.shape[1]-450, img.shape[0]]])
    # get transformation matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    # get inverse transformation matrix invM
    invM = cv2.getPerspectiveTransform(dst, src)
    # create warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return (M, invM, warped)

def mask(img, vertices):
    """Applies an image mask. Only keeps the region of the image defined by the polygonformed from `vertices`. 
    The rest of the image is set to black.
    
    Keyword arguments:
    img -- input image
    vertices -- vertices that defines a region
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def warped_binary(img, objpoints, imgpoints, lower_yellow, upper_yellow, lower_white, upper_white, 
                  sobel_x_threshold=[30, 50], ang_threshold=[0.9, 1.1], kernel_size=7):
    '''Undistort, threshold and warp an input image
    
    Keyword arguments:
    img -- input image
    objpoints -- 3d points in real world space
    imgpoints -- 2d points in image plane
    lower_yellow -- lower threshold for yellow
    upper_yellow -- upper threshold for yellow
    lower_white -- lower threshold for white
    upper_white -- upper threshold for white    
    sobel_x_threshold -- threshold for x gradient
    ang_threshold -- threhold for gradient direction
    kernel_size -- sobel operator kernel size
    '''
    img = np.copy(img)
    # undistort
    undist = cal_undistort(img, objpoints, imgpoints)
    
    # mask
    #masked_img = mask(undist, vertices)
    # threshold
    bin_img = binary_img(img, lower_yellow, upper_yellow, lower_white, upper_white,
                         sobel_x_threshold, ang_threshold, kernel_size)
    # draw region
    #region = draw_region(bin_img, v1, v2, v3, v4)
    # mask
    masked_bin = mask(bin_img, mask_vertices)
    # warp
    M, invM, warped = warp(masked_bin)
    return M, invM, warped, undist


def sliding_window_search(binary_warped, nwindows, margin, minpix, visualize = False):
    '''Find lanes in a binary warped image
    
    Keyword arguments:
    binary_warped -- input binary image and already warped into top view
    nwindows -- number of sliding windows
    margin -- width of windows, +/- margin
    minpix -- minimum number of pixels found to recenter the window
    visualize -- boolean value to turn on visualization, for testing only
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    #nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    #margin = 100
    # Set minimum number of pixels found to recenter window
    #minpix = 5
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if visualize:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit = np.polyfit(lefty, leftx, 2)
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if visualize: 
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    return ploty, left_fitx, right_fitx, out_img, lefty, leftx, righty, rightx

def visualize_lane(undist, binary_warped, lane, invM):
    ''' Visualize lanes on an undistorted image.
    
    Keyword arguments:
    undist -- an undistorted image
    binary_warped -- a binary warped image
    lane -- [ploty, left_fitx, right_fitx]
    invM -- inverse transform matrix to unwarp image
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = lane[0]
    left_fitx = lane[1]
    right_fitx = lane[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw area inside the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (invM)
    newwarp = cv2.warpPerspective(color_warp, invM, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result) 
    return result

def process_lane_data(lane, img_shape, verbose=False):
    """Process left and right lane fitted data from binary image
    
    Return radius of curvature, offset from lane center, derivative of lanes at bottom max y
    
    keyword arguments:
    lane -- [ploty, left_fitx, right_fitx]
    img_shape -- [height, width]
    verbose -- debug control
    
    """
    ploty = lane[0]
    left_fitx = lane[1]
    right_fitx = lane[2]
    # evaluate curvature at the bottom of the image
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate derivatives for parallelism check
    left_dot = 2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1]  
    right_dot = 2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1]
    
    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + left_dot**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + right_dot**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Compute left and right lane at bottom of image
    left_bot_x = left_fitx[np.argmax(ploty)]
    right_bot_x = right_fitx[np.argmax(ploty)]
    
    # Compute lane center
    lane_center = (right_bot_x + left_bot_x) / 2
    # Compute camera location, assuming camera is mounted at center of vehicle
    camera_x = img_shape[1] / 2
    # Compute lateral offset, if offset > 0, vehicle deviates to the right, otherwise deviates to the left
    offset = camera_x - lane_center
    # Convert to real world unit
    offset = offset*xm_per_pix
    
    # Print for debugging
    if verbose:
        print("Left Lane Radius: {0:.2f} m, Right Lane Radius: {1:.2f} m" .format(left_curverad, right_curverad))
        if offset < 0:
            print("Offset: {:.2f} m left".format(offset))
        else:
            print("Offset: {:.2f} m right".format(offset))
    
    return left_curverad, right_curverad, left_dot, right_dot, offset

def quick_search(binary_warped, margin, left_fit, right_fit):
    # select detected pixels
    nonzero = binary_warped.nonzero()
    # initialize x and y array
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # search for left lane pixels, x_old - margin <= x_new <= x_old + margin
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    # repeat for right lane pixels
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]    
    return ploty, left_fitx, right_fitx, out_img, lefty, leftx, righty, rightx

def lane_validated(ploty, left_fitx, right_fitx, left_curverad, right_curverad, left_dot, right_dot, verbose=False):
    flag = True
    # check radius of curvature
    if left_curverad / right_curverad > 2 or left_curverad / right_curverad < 0.5:
        flag = False
        if verbose:
            print("Radius ratio", left_curverad / right_curverad)
            print("Radius check failed")
        return flag
    # check lane width, 300 pixels < lane width < 400 pixels
    left_bot_x = left_fitx[np.argmax(ploty)]
    right_bot_x = right_fitx[np.argmax(ploty)]
    if right_bot_x - left_bot_x > 400 or right_bot_x - left_bot_x < 300:
        flag = False
        if verbose:
            print("Lane width", right_bot_x - left_bot_x, "pixels")
            print("Lane width check failed")
        return flag
    # check parallelism
    if np.absolute(left_dot / right_dot) > 10 or np.absolute(left_dot / right_dot) < 0.1:    
        flag = False
        if verbose:
            print("Derivative ratio", left_dot / right_dot)
            print("Parallelism check failed")
        return flag
    return flag

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last 3 fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last 3 iterations
        self.bestx = []     
        # polynomial coefficients averaged over the last 3 iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.offset = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
    
    def update(self, ploty, fitx, curverad, dot, offset):
        """Use newly detected lane to update left and right lane object, once they pass the validation check

        Keyword arguments:
        lane -- a lane object, left or right
        ploty -- y values for fitting, returned from sliding window or quick search
        fitx -- x values for fitting, returned from sliding window or quick search
        curverad -- radius of curvature
        dot -- derivative of curve at bottom of the image
        offset -- offset from center of lane
        """
        self.detected = True
        if len(self.recent_xfitted) == 3:
            # delete oldest result
            self.recent_xfitted.pop(0)
        self.recent_xfitted.append(fitx)
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(ploty, self.bestx, 2)
        self.current_fit = np.polyfit(ploty, fitx, 2)
        self.radius_of_curvature = curverad
        self.offset = offset
        self.diffs = self.current_fit - self.best_fit
        self.allx = fitx
        self.ally = ploty
    
    def plot_x(self):
        plot_x = self.best_fit[0]*self.ally**2 + self.best_fit[1]*self.ally + self.best_fit[2]
        return plot_x

def add_text(lane_img, left_curverad, right_curverad, offset, validated):
    """Add useful information to output image
    
    Keyword arguments:
    lane_img -- output image with lanes visualized
    left_curverad -- radius of left lane line
    right_curverad -- radius of right lane line
    offset -- offset of vehicle from center of lane
    validated -- flag, false if detection fails in last frame
    """
    pos = "left"
    status = "Current"
    if offset > 0:
        pos = "right"
    if not validated:
        status = "Last"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_left = "Left Radius: " + str(left_curverad) + " m"
    text_right = "Right Radius: " + str(right_curverad) + " m"
    text_offset = "Offset: " + str(np.absolute(offset)) + " " + pos 
    text_status = "Using " + status + " detection"
    left_pos = (100, 100)
    right_pos = (100, 150)
    offset_pos = (100, 200)
    status_pos = (100, 250)
    font_size = 1
    font_color = (255, 255, 0)
    text_thickness = 2
    cv2.putText(lane_img, text_left, left_pos, font, font_size, font_color, text_thickness)
    cv2.putText(lane_img, text_right, right_pos, font, font_size, font_color, text_thickness)
    cv2.putText(lane_img, text_offset, offset_pos, font, font_size, font_color, text_thickness)
    cv2.putText(lane_img, text_status, status_pos, font, font_size, font_color, text_thickness)
    return lane_img

def lane_detection_pipeline(img, objpoints, imgpoints, lower_yellow, upper_yellow, lower_white, upper_white, 
             sobel_x_threshold, ang_threshold, kernel_size, left_lane, right_lane):
    # get warped binary image
    M, invM, bin_warped, undist = warped_binary(img, objpoints, imgpoints, lower_yellow, upper_yellow, lower_white, upper_white, 
                                 sobel_x_threshold, ang_threshold, kernel_size)
    if (not left_lane.detected) or (not right_lane.detected) :
        # lanes are not detected in last frame, use sliding window search
        #print("Searching")
        ploty, left_fitx, right_fitx, out_img, lefty, leftx, righty, rightx = \
            sliding_window_search(bin_warped, nwindows, sliding_margin, minpix, visualize = False)
    else:
        #print("Quick searching")
        # last detection is valid, use quick search
        ploty, left_fitx, right_fitx, out_img, lefty, leftx, righty, rightx = \
            quick_search(bin_warped, sliding_margin, left_lane.current_fit, right_lane.current_fit)
    # proecess lane data for validation purposes    
    left_curverad, right_curverad, left_dot, right_dot, offset = \
        process_lane_data([ploty, left_fitx, right_fitx], bin_warped.shape, False)
    # update lanes if detection is valid
    validated = lane_validated(ploty, left_fitx, right_fitx, left_curverad, right_curverad, left_dot, right_dot, verbose=False)
    if validated or left_lane.best_fit == None or right_lane.best_fit == None:
        # update lane if detection is valid or if no detection is present
        left_lane.update(ploty, left_fitx, left_curverad, left_dot, offset)
        right_lane.update(ploty, right_fitx, right_curverad, right_dot, offset)
    else:
        # set detected to None
        left_lane.detected = False
        right_lane.detected = False
    # use line object member to visualize 
    l_plot_x = left_lane.plot_x()
    r_plot_x = right_lane.plot_x()
    #debug
    #print(l_plot_x.shape)
    #print(r_plot_x.shape)
    visual = visualize_lane(undist, bin_warped, [left_lane.ally, l_plot_x, r_plot_x], invM)
    # add text to image
    visual = add_text(visual, left_lane.radius_of_curvature, right_lane.radius_of_curvature, left_lane.offset, validated)
    return visual

