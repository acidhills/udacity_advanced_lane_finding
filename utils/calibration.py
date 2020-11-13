import numpy as np
import cv2

def get_object_points_(img,nx = 9,ny = 6):

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
        ret ,objp,imgp = get_object_points_(img)
        if(ret):
            object_points.append(objp)
            image_points.append(imgp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[:2],None,None) 
    return mtx, dist, rvecs, tvecs, object_points, image_points

def getPerspectiveTransformMatrix():    
    src = np.float32([[500,500], [800,500],[250,700], [1150,700]])
    dst = np.float32([[400,0],[1050,0],[400,700],[1050,700]])
    
    src = np.float32([[500,500], [800,500],[250,700], [1150,700]])
    dst = np.float32([[400,0],[1050,0],[400,700],[1050,700]])

    xlen, ylen = 1280,720

    road = 600

    road_narrow = road *0.33
    road_wide = road*1.5
    yk = 0.67

    src = np.int32([[(xlen/2) -road_narrow/2 ,ylen*yk],
                    [(xlen/2) +road_narrow/2 ,ylen*yk],
                    [(xlen/2) +road_wide/2, ylen],
                    [(xlen/2) -road_wide/2, ylen]
                   ])
    
    dst = np.int32([[(xlen/2) -road/2 ,0],
                    [(xlen/2) +road/2 ,0],
                    [(xlen/2) +road/2, ylen],
                    [(xlen/2) -road/2, ylen]])
    src = np.float32(src)
    dst = np.float32(dst)
#     print(src,'\n',dst)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv