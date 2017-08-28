## Writeup Template

---

**Advanced Lane Finding Project**

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/cheess.png
[image2]: ./images/test3.png
[image3]: ./images/test3_undist.png
[image4]: ./images/threshold1.png
[image4.2]: ./images/threshold2.png
[image5]: ./images/perspective.png
[image6]: ./images/hist.png
[image7]: ./images/lane_detect.png
[image8]: ./images/lane.png
[image9]: ./images/merged.png
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

### Camera Calibration

#### 1. Briefly, state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 113 through 134 of the file called `pipeline_run.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
The chessboard pattern calibration images (in the `cal_images` folder) contain 9 and 6 corners in the horizontal and vertical directions, respectively.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![chessboard pattern][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Using `cam_mtx` and `cam_dist` I undistorted the images with the function `cv2.undistort`:
`img = cv2.undistort(np.copy(img), self.cam_mtx, self.cam_dist)`
Here I show distorted and undistorted
![distorted][image2]
![undistorted][image3]

#### 2. Thresholding
I tried several combinations of color and gradient transforms and also after some research and reading I found OpenCV has a method already implemented that can greatly help with this problem. I used the Laplacian function implemented on `cv2.Laplacian`.

First I applied thresholding defined on function `threshold` lines 79 to 111 of `pipeline_run.py` then I applied the Laplacian function and more thresholding to highlight only negative values. This was the best performing method I found.
The Laplacian part is defined on function `find_edges` on lines 195 to 217 of `pipeline_run.py`
Here's an example of my output for this step after the first thresholding and the second one.
In this case, the two thresholding stages work the same but in other frames, the second stage helped to get a better lane detection.
![thresholded][image4]
![thresholded2][image4.2]

#### 3. Perspective Transform.

The code for my perspective transform is defined on the functiion `transform_perspective` on lines 219 to 256 of `pipeline_run.py`. I hardcoded the destination of the perspective transform as:
```python
dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                [1000./1280.*img_size[1], 100./720.*img_size[0]],
                [1000./1280.*img_size[1], 720./720.*img_size[0]],
                [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
```
But I used the function `find_perspective_points` defined on lines 137 to 193 of `pipeline_run` where I calculate `src` using part of the code for project 1.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![perspective][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Then after warping the image with `cv2.warpPerspective` I used the histogram method for detecting the right and left lanes. This is implemented on lines 26 to 122 on file `utils.py`. I spent some time tuning the size of the sliding window and the minpix parameter.
At the end of this function after recentering the windows if I found more than 15 pixels I used the `np.polyfit` function for fitting a second order polynomial.
The last step here was to draw the shaded area for the lane.
![histogram][image6]
![image with poly][image7]
![shaded area][image8]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this on the function `add_figures_to_image` defined in lines 258 to 251 of my code in `pipeline_run.py`.
First I calculated the curvature of the left and right lanes and averaged it. I also calculated the minimum curvature between the two lanes and the vehicle position.
Then I converted this values into meters because the calculated curvature and position are on pixel value. I used  this conversion:
```
# Convert from pixels to meters
vehicle_position = vehicle_position / 12800 * 3.7
curvature = curvature / 128 * 3.7
    min_curvature = min_curvature / 128 * 3.7
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After I got the lanes and the drawn area of the detected lane, I warped back the perspective using the `warp_minv` that I saved earlier when I warped the perspective.
And after warping back the lane image I added it to the original image with: `img = cv2.add(img, thi)`. This is defined from line 58 to 63 of `pipeline_run.py'.
I used the function `highlight_lane_line_area` defined on line 147 from `utils.py` to draw the area.

And lastly, I plotted the curvature and center offset information onto the image. I calculated the curvature and the things described on the last point here.

Here is the final image: 
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

The pipeline works reasonably well, certainly much better than the first project and looks cool also. It didn't work that well for the challenge video. I think the major problem on the challenge video was the lack of contrast and false lines. Maybe some histogram equalization or normalizing the image before the pipeline could help with this problem of illumination variations.

The most tricky part of the entire project was the sliding window techniques, also there are a lot of parameters to manually tune on all the steps of the image processing pipeline, I wonder if we could use a machine learning algorithm to get the best parameters for each particular video.

I think there are still improvements needed on the thresholding because it's very susceptible to lighting changes, also the pipeline is a little bit slow and it would be better to parallelize the process or use a GPU implementation of OpenCV.enCV.
