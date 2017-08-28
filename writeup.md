## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

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

First I applied thresholding defined on function `threshold` lines 79 to 111 of `pipeline_run.py` then I applied the laplacian function and more thresholding to highlight only negative values. This was the best performing method I found.
The laplacian part is defined on function `find_edges` on lines 195 to 217 of `pipeline_run.py`
Here's an example of my output for this step.

![thresholded][image4]

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
![image with poly][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I did this on the function `add_figures_to_image` defined in lines 258 to 251 of my code in `pipeline_run.py`

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
