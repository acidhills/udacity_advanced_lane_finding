import cv2
import numpy as np

def abs_sobel_thresh(img, orient='x', thresh=(0, 255),sobel_kernel=3,):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = None
    if(orient == 'x'):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    direction = np.arctan2(sobely, sobelx)
    
    binary = np.zeros_like(direction)
    binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary

def filter_with_hough(binary):
    output = np.zeros_like(binary)
    hough = cv2.HoughLinesP(binary, 1, (np.pi/180)*1, 1, np.array([]), minLineLength=8, maxLineGap=5)
    for x1,y1,x2,y2 in [x[0] for x in hough]:
        cv2.line(output,(x1,y1),(x2,y2),(255,255,255),2)
    return output


def color_treshold(binary, treshold=(170,255)):
    thresh_min = treshold[0]
    thresh_max = treshold[1]
    r_binary = np.zeros_like(binary)
    r_binary[(binary >= thresh_min) & (binary <= thresh_max)] = 1
    return r_binary
    
def get_tresholded_img(image):
    ksize = 15 
    gkernel_size = 17
    image = cv2.GaussianBlur(image, (gkernel_size, gkernel_size), 0)
#     return image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
#     a = lab[:,:,1]
#     b = lab[:,:,2]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
#     h_channel = hls[:,:,0]
    l_channel = cv2.equalizeHist(l_channel)
    s_channel = cv2.equalizeHist(s_channel)
#     h_channel = cv2.equalizeHist(s_channel)
    gray = cv2.equalizeHist(gray)
    
    L = cv2.equalizeHist(L)
    
    working_channel = l_channel + s_channel
#     mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, mag_thresh=(25, 100))
#     s_binary = color_treshold(s_channel, (170,255))
#     gradx = abs_sobel_thresh(working_channel, orient='x', sobel_kernel=ksize, thresh=(25, 100))
#     gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25, 100))
#     grady = abs_sobel_thresh(working_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

    gradx = abs_sobel_thresh(working_channel, orient='x', sobel_kernel=ksize, thresh=(25, 100))
    
    binary_l = color_treshold(L, (200,255))
    
    combined = np.zeros_like(s_channel)
#     (s_binary == 1) |
    combined[ (binary_l == 1)| (gradx*dir_binary == 1)] = 1
    combined_image = np.array(combined * 255, dtype = np.uint8)
    return combined_image