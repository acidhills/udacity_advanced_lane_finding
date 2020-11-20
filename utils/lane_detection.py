import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_lane_pixels_(binary_warped, nwindows = 9, margin = 50, minpix = 50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

   
    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []
    rectangels = []
    
    for window in range(nwindows):
        
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
         
        rectangels.append(((win_xleft_low,win_y_low),(win_xleft_high,win_y_high)))
        rectangels.append(((win_xright_low,win_y_low),(win_xright_high,win_y_high)))
               
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
      
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))


    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, rectangels


def get_fit_(left_fit, right_fit, binary_warped):    
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fitx,right_fitx,ploty

def fill_poly_(image,left_fitx,right_fitx,ploty, color):
    points_left = zip(left_fitx,ploty)   
    points_right = zip(right_fitx,ploty)
    points_left = [[x,y] for x,y in points_left]
    points_right = [[x,y] for x,y in points_right]
    points_right = points_right[::-1]
    points = np.concatenate((points_left, points_right))
    points = np.int32(points)
    cv2.fillPoly(image, [points], color=color)
    
def get_lanes(binary_warped, debug = False):
    leftx, lefty, rightx, righty, rectangels = find_lane_pixels_(binary_warped)
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    
    left_fitx,right_fitx,ploty = get_fit_(left_fit,right_fit,binary_warped)
    
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    if debug:
        for low, high in rectangels:        
            cv2.rectangle(out_img,low,high,(0,255,0), 2) 
        
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
    
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    else:
        out_img[:,:] = [0,0,0,]
        fill_poly_(out_img,left_fitx,right_fitx,ploty,[0, 255, 0])
    
#     out_img[lefty, leftx] = [255, 0, 0]
#     out_img[righty, rightx] = [0, 0, 255]
#     out_img[np.int32(ploty),np.int32(left_fitx)] = [0,177,228]
#     out_img[np.int32(ploty),np.int32(left_fitx) + 2] = [0,177,228]
#     out_img[np.int32(ploty),np.int32(left_fitx) + 1] = [0,177,228]
#     out_img[np.int32(ploty),np.int32(left_fitx) - 1] = [0,177,228]
#     out_img[np.int32(ploty),np.int32(left_fitx) - 2] = [0,177,228]
#     top = ([left_fit[0]*0**2 + left_fit[1]*0 + left_fit[2],0])
#     cv2.rectangle(out_img, , ,(0,255,0), 2)
#     out_img[np.int32(ploty),np.int32(right_fitx)] = [0,0,255]
    return out_img