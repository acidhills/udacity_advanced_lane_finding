import numpy as np
import cv2

def get_object_points(img,nx = 9,ny = 6):

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     plt.imshow(gray, cmap='gray')
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) 
    if ret:
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        return True,objp,corners
    else:
        return False,[], []
    
def clalibrate_camera(imgs):
    object_points = []
    image_points = []
    for img in imgs:
        ret ,objp,imgp = get_object_points(img)
        if(ret):
            object_points.append(objp)
            image_points.append(imgp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[:2],None,None) 
    return mtx, dist, rvecs, tvecs