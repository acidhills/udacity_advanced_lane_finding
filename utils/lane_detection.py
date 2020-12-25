import numpy as np
import cv2
import matplotlib.pyplot as plt


def curvature(left_fit, right_fit,img):
    def curv_(x,y):
        ym_per_pix = 30/720 
        xm_per_pix = 3.7/700 
        x_ = x*xm_per_pix
        y_ = y *  ym_per_pix
        fit = np.polyfit(y_,x_,2) 

        A = fit[0]
        B = fit[1]
        y_ = np.max(y_)
        r = ((1 +(2*A*y_+B)**2)**1.5)/np.absolute(2*A)
        return r
    
    left_fitx,right_fitx,ploty = get_fit_(left_fit, right_fit,img)
    l_curv = curv_(left_fitx,ploty)
    r_curv = curv_(right_fitx,ploty)
    return np.mean([l_curv,r_curv])
    
    

def hist(img,hist_window = 5, y_k = 2):
    ylen = int(img.shape[0] / y_k)
    
    bottom_half = img[ylen:,:]
    
    histogram = np.sum(bottom_half, axis=0)
    histogram = [np.sum(histogram[dot-hist_window:dot+hist_window]) for dot in np.arange(hist_window, len(histogram) - hist_window)]
#     histogram =[np.mean(histogram[dot-hist_window:dot+hist_window]) for dot in np.arange(hist_window, len(histogram) - hist_window)]

    histogram = np.array(histogram)
    
    return histogram

def get_r_and_l_mid_(img):
    
    hist_window = 5
    histogram = hist(img,hist_window)
    deep_hist = hist(img,hist_window,10)
    
    midpoint = np.int(histogram.shape[0]//2)
    leftx_weight = histogram[np.argmax(histogram[:midpoint])]
    rightx_weight = histogram[midpoint+np.argmax(histogram[midpoint:])]
    
    leftx_base = np.argmax(histogram[:midpoint]) +hist_window
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint + hist_window
    if(leftx_weight > (3*rightx_weight)):
        r_hist = deep_hist[midpoint:]
        rightx_base = np.argmax(r_hist)+ midpoint + hist_window
    if(rightx_weight > (3*leftx_weight)):
        l_hist = deep_hist[:midpoint]
        leftx_base = np.argmax(l_hist)+ hist_window
    
    return leftx_base,rightx_base
    
    

def search_around_poly_(binary_warped,left_fit,right_fit, margin = 100):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = None
    right_lane_inds = None
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def find_lane_pixels_(binary_warped,nwindows = 9, margin = 100, minpix = 50):
    hist_img = binary_warped
    hist_window = 5    
    histogram = hist(hist_img,hist_window)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint]) +hist_window
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint + hist_window
    
    leftx_base,rightx_base = get_r_and_l_mid_(hist_img)
   
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

    return leftx, lefty, rightx, righty, rectangels#,leftx_base,rightx_base


def get_fit_(left_fit, right_fit, binary_warped):    
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ploty = ploty[int(binary_warped.shape[0]*0.3):]
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

class LaneDetector:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.i = 0
        self.recalc_counter = 0
        self.fits_changed = 0
    
    def print_stats(self):
        print("Recalcs: {}, Fits_changed: {}".format(self.recalc_counter, self.fits_changed))
         
    def get_lanes(self, binary_warped,debug = False, enable_drop_fix = True, enable_smothing = True, smooth_k = 0.7):
        rectangels = None
        if (not enable_smothing) or (self.i%10 == 0) or (self.left_fit is None) or (self.right_fit is None):
            self.recalc_counter+=1
            leftx, lefty, rightx, righty, rectangels = find_lane_pixels_(binary_warped)
        else:
            leftx, lefty, rightx, righty = search_around_poly_(binary_warped,self.left_fit,self.right_fit)
        if len(leftx)!=0:
            new_left_fit = np.polyfit(lefty,leftx,2) 
        else:
            new_left_fit = self.left_fit
        
        if len(rightx)!=0:
            new_right_fit = np.polyfit(righty,rightx,2)
        else:
            new_right_fit = self.right_fit
        
        
        if(self.left_fit is None or self.right_fit is None):
            self.left_fit = new_left_fit
            self.right_fit = new_right_fit
#         else:

        left_diff = abs(self.left_fit[2] - new_left_fit[2])
        right_diff = abs(self.right_fit[2] - new_right_fit[2]) 
        fit_diff = (new_right_fit[2]-new_left_fit[2])
        if((not enable_drop_fix) or(fit_diff < 600 and fit_diff>300)):
            self.fits_changed += 1
            k = smooth_k
            self.left_fit = k*self.left_fit + (1-k)* new_left_fit
            self.right_fit = k*self.right_fit + (1-k)* new_right_fit
        
        
        
        curv = curvature(self.left_fit,self.right_fit, binary_warped)
            
        left_fitx,right_fitx,ploty = get_fit_(self.left_fit,self.right_fit,binary_warped)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        if debug:
            if rectangels is not None:
                for low, high in rectangels:        
                    cv2.rectangle(out_img,low,high,(0,255,0), 2) 

            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
        else:
            out_img[:,:] = [0,0,0,]
            fill_poly_(out_img,left_fitx,right_fitx,ploty,[0, 255, 0])
        self.i += 1
        return out_img,curv