**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[video1]: ./project_video.mp4 "Video"
[orginal]: ./images/calibration.jpg
[undistorted]: ./images/calibration_undist.jpg
[orginal2]: ./images/original.jpg
[undistorted2]: ./images/undistorted.jpg
[tresholding]: ./images/tresholding.jpg
[perspective_unwarped]: ./images/perspective_unwarped.jpg
[perspective_warped]: ./images/perspective_warped.jpg
[lane_search]: ./images/lane_search.jpg
[result]: ./images/result.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


### Writeup / README

### Camera Calibration

The code for this step is contained in the './utils/calibration.py' clalibrate_camera function. An example of usage you can find in `project.ipynb`, cell 16, the constructor of Processor class. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![original_image][orginal]
![undistorted_image][undistorted]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
<br>
![original][orginal2]
![undistorted_image][undistorted2]

#### 2. Color transforms, gradients or other.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function 'get_tresholded_img()' in `tresholding.py`). 
Speaking exactly, I used `cv2.GaussianBlur()` to reduce the count of small particles, `cv2.equalizeHist` for all channels to normalize them. Then I used s_chanel from HLS with direction Sobel operator, a combination of S and L channels from HLS with x oriented Sobel and L channel from LAB color space with color thresholding. Also, I had tried to use hofflines to reduce the count of particles but after several experiments deleted this step. 
All these actions produced the next result:
![tresholded image][tresholding]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `getPerspectiveTransformMatrix()`, which appears in the file `./utils/calibration.py`.  The `getPerspectiveTransformMatrix()` function takes as inputs an image (`img`).  I chose the next algorithm for obtaining the source and the destination points:

```
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
    
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
<br>
![alt perspective_unwarped][perspective_unwarped]
![alt perspective_warped][perspective_warped]


#### 4. Lane identification and fitting,

The line detection algorithm lies in `utils/lane_detection.py`, `LaneDetector` class, `get_lanes` method. I used the provided course algorithm with several improvements. First of all, I added a sliding window, histogram which sums pixel count for each 5 neighborhood dots, which should help us to choose shortlines against particles clouds. Next, I choose the less obvious side(the side with smaller detected dost) and search line in it with the higher threshold, which again helps us to detect short lines.
Also, I added several checks, which helps us to avoid using lanes with a too wide or thin gap between them, and added a standard smoothing technic, which uses only 30% of new line coefficients. 
And, at last, I use lightweight search around poly function, but to avoid too big errors I make full recalculations every 10 frames.

Additionally, not during the lane search, but just before it, inside the Processor class, `project.ipynb`, cell 16, before this step and after previous, I cut 200 pixels from the left and right sides of the image. 

Here is the result:
<br>
![lane_search][lane_search]

#### 5. Curvature.
This is an easy part, i just use standart formula ((1 + (2 * A * y +B)^2)^1.5)/absolute(2 * A) where A and B is polynomial coefficients from formula y = A * x^2 + B * x + C. The only interesting thing is, that I refit identified lane points with respect to real-world distance coefficients.
All this staff is located in `utils/lane_detection.py` file, the `curvature` function, and it used inside `LaneDetector` class.

#### 6. Example of an image.

The whole pipeline is implemented inside `Processor` class in `project.ipynb` notebook, cell 16.  Here is an example of my result on a test image:
<br>
![result][result]

---

### Pipeline (video)


Here's a [link to my video result](./project_video_out.mp4)

### Discussion

There several issues with the pipeline, but the most annoying is the wrong fit, I mean, when you totally correct found the lines, and after fit got too curved line. The only idea to solve this issue is to use previous project technic, and find several hogh lines and fit only them. Also, I think there are many problems with performance( from 4 to 6 frames per second) and, I still have some issues with lanes in shadows(under the bridge, for example). I guess I could solve them after more experiments with colorspaces.