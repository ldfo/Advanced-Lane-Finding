import os
import cv2
import glob
import numpy as np
from math import *
# import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import collections
from itertools import chain
from functools import reduce
from scipy.signal import find_peaks_cwt
from moviepy.editor import VideoFileClip
from utils import histogram_pixels, get_pos
# from lanes import Lane

class Pipeline(object):
    """Pipeline that processes video."""
    def __init__(self,
                in_file,
                out_file,
                calibration_files):
        self.cam_mtx, self.cam_dist = self.get_cal_and_dist(calibration_files)
        print ('calibration matrix: ', self.cam_mtx)
        print ('cam dist:', self.cam_dist)
        print ('loading:', in_file)
        in_clip = VideoFileClip(in_file)
        # in_clip = in_clip.subclip(10,15)
        print ('processing:', in_file)
        processed_clip = in_clip.fl_image(self.process_image)
        print ('writing to:', out_file)
        processed_clip.write_videofile(out_file, audio=False, threads=8)
        # for testing with single image
        # image = mpimage.imread('test_images/test3.jpg')
        # image = self.process_image(image)
        # plt.imshow(image)
        # plt.show()

    def process_image(self, img):
        img_size = (img.shape[1], img.shape[0])
        width, height = img_size
        # undistort image
        img = cv2.undistort(np.copy(img), self.cam_mtx, self.cam_dist)
        # thresholded binary image.
        thi = self.threshold(img, color=False)
        thi = np.dstack((thi, thi, thi)) * 255
        # perspective transform
        src = self.find_perspective_points(thi)
        warp_m, warp_minv = self.transform_perspective(thi, src)
        thi = cv2.warpPerspective(thi, warp_m,
                                 (thi.shape[1],
                                  thi.shape[0]),
                                  flags=cv2.INTER_LINEAR)
        # get lane mask
        thi, left_fit, right_fit = histogram_pixels(thi)
        # mix lane mask with original image
        # first transform it back to org perspective
        thi = cv2.warpPerspective(thi, warp_minv,
                                 (thi.shape[1],
                                  thi.shape[0]),
                                  flags=cv2.INTER_LINEAR)
        # add mask to original image
        img = cv2.add(img, thi)
        # add figures to image
        self.add_figures_to_image(img, left_fit, right_fit)
        return img

    def fit_lines(self, img):
        leftx, lefty, rightx, righty = histogram_pixels(img)
        # Fit a second order polynomial to each fake lane line
        left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
        print("Left coeffs:", left_coeffs)
        print("righty[0]: ,", righty[0], ", rightx[0]: ", rightx[0])
        right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)
        print("Right coeffs: ", right_coeffs)
        polyfit_left = draw_poly(blank_canvas, lane_poly, left_coeffs, 30)
        polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 30)
        return polyfit_drawn

    def threshold(self, img, color=False, mag_dir_thresh=False):
        """Threshhold image on saturation channel and
        using magnitude gradient"""
        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)


        ## White Color
        lower_white = np.array([0,210,0], dtype=np.uint8)
        upper_white = np.array([255,255,255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        ## Yellow Color
        lower_yellow = np.array([18,0,100], dtype=np.uint8)
        upper_yellow = np.array([30,220,255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        combined_binary = np.zeros_like(white_mask)

        # Dir Mag Threshold
        if mag_dir_thresh:
            dir_mask = dir_threshold(img)
            mag_mask = mag_thresh(img)
            combined_binary[((dir_mask == 1) & (mag_mask == 1))] = 255

        if color:
            return np.dstack((white_mask, yellow_mask, combined_binary))

        else:
            combined_binary[((white_mask == 255) | (yellow_mask == 255))] = 255
            combined_binary[(combined_binary == 255)] = 1
            return combined_binary

    def get_cal_and_dist(self, calibration_path):
        cal_images = glob.glob(calibration_path)
        nx, ny = 9, 6
        objpoints = []
        imgpoints = []

        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
        # for each file in calibration images
        for fname in cal_images:
            # read image
            img = cv2.imread(fname)
            # to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # calibrate camera
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        return mtx, dist

    def find_perspective_points(self, image):
        edges = self.find_edges(image)

        # Computing perspective points automatically
        rho = 2              # distance resolution in pixels of the Hough grid
        theta = 1*np.pi/180  # angular resolution in radians of the Hough grid
        threshold = 100       # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100 # minimum number of pixels making up a line
        max_line_gap = 25    # maximum gap in pixels between connectable line segments

        angle_min_mag = 20*pi/180
        angle_max_mag = 65*pi/180

        lane_markers_x = [[], []]
        lane_markers_y = [[], []]

        masked_edges = np.copy(edges)
        masked_edges[:edges.shape[0]*6//10,:] = 0
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                theta = atan2(y1-y2, x2-x1)
                rho = ((x1+x2)*cos(theta) + (y1+y2)*sin(theta))/2
                if (abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag):
                    if theta > 0: # positive theta is downward in image space?
                        i = 0 # Left lane marker
                    else:
                        i = 1 # Right lane marker
                    lane_markers_x[i].append(x1)
                    lane_markers_x[i].append(x2)
                    lane_markers_y[i].append(y1)
                    lane_markers_y[i].append(y2)

        if len(lane_markers_x[0]) < 1 or len(lane_markers_x[1]) < 1:
            # Failed to find two lane markers
            return None

        p_left  = np.polyfit(lane_markers_y[0], lane_markers_x[0], 1)
        p_right = np.polyfit(lane_markers_y[1], lane_markers_x[1], 1)

        # Find intersection of the two lines
        apex_pt = np.linalg.solve([[p_left[0], -1], [p_right[0], -1]], [-p_left[1], -p_right[1]])
        top_y = ceil(apex_pt[0] + 0.075*edges.shape[0])

        bl_pt = ceil(np.polyval(p_left, edges.shape[0]))
        tl_pt = ceil(np.polyval(p_left, top_y))

        br_pt = ceil(np.polyval(p_right, edges.shape[0]))
        tr_pt = ceil(np.polyval(p_right, top_y))

        src = np.array([[tl_pt, top_y],
                        [tr_pt, top_y],
                        [br_pt, edges.shape[0]],
                        [bl_pt, edges.shape[0]]], np.float32)

        return src

    def find_edges(self, image, mask_half=False):
        hls = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
        s = hls[:,:,2]
        gray = (0.5*image[:,:,0] + 0.4*image[:,:,1] + 0.1*image[:,:,2]).astype(np.uint8)

        _, gray_binary = cv2.threshold(gray.astype('uint8'), 130, 255, cv2.THRESH_BINARY)

        # switch to gray image for laplacian if 's' doesn't give enough details
        total_px = image.shape[0]*image.shape[1]
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
        mask_one = (laplacian < 0.15*np.min(laplacian)).astype(np.uint8)
        if cv2.countNonZero(mask_one)/total_px < 0.01:
            laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
            mask_one = (laplacian < 0.075*np.min(laplacian)).astype(np.uint8)

        _, s_binary = cv2.threshold(s.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
        mask_two = s_binary


        combined_binary = np.clip(cv2.bitwise_and(gray_binary,
                            cv2.bitwise_or(mask_one, mask_two)), 0, 1).astype('uint8')

        return combined_binary

    def transform_perspective(self, image, src_in = None, dst_in = None, display=False):

        img_size = image.shape
        if src_in is None:
            src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
                            [705. /1280.*img_size[1], 455./720.*img_size[0]],
                            [1130./1280.*img_size[1], 720./720.*img_size[0]],
                            [190. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
        else:
            src = src_in

        if dst_in is None:
            dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                            [1000./1280.*img_size[1], 100./720.*img_size[0]],
                            [1000./1280.*img_size[1], 720./720.*img_size[0]],
                            [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
        else:
            dst = dst_in

        warp_m = cv2.getPerspectiveTransform(src, dst)
        warp_minv = cv2.getPerspectiveTransform(dst, src)

        if display:
            plt.subplot(1,2,1)
            plt.hold(True)
            plt.imshow(image, cmap='gray')
            colors = ['r+','g+','b+','w+']
            for i in range(4):
                plt.plot(src[i,0],src[i,1],colors[i])

            im2 = cv2.warpPerspective(image, warp_m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            plt.subplot(1,2,2)
            plt.hold(True)
            plt.imshow(im2, cmap='gray')
            for i in range(4):
                plt.plot(dst[i,0],dst[i,1],colors[i])
            plt.show()
        return warp_m, warp_minv

    def add_figures_to_image(self,img, left_fit, right_fit):
        """
        Draws information about the center offset and the current lane curvature onto the given image.
        :param img:
        """
        # Calculate curvature
        y_eval = 500
        left_curverad = np.absolute(((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2) ** 1.5) \
                        /(2 * right_fit[0]))
        right_curverad = np.absolute(((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) \
                         /(2 * right_fit[0]))
        curvature = (left_curverad + right_curverad) / 2
        min_curvature = min(left_curverad, right_curverad)
        vehicle_position = get_pos(719, left_fit, right_fit)
        # Convert from pixels to meters
        vehicle_position = vehicle_position / 12800 * 3.7
        curvature = curvature / 128 * 3.7
        min_curvature = min_curvature / 128 * 3.7

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = "left" if vehicle_position < 0 else "right"
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)
        cv2.putText(img, 'Min Radius of Curvature = %d(m)' % min_curvature, (50, 150), font, 1, (255, 255, 255), 2)

if __name__ == '__main__':
    # calibrate camera
    pipe = Pipeline(in_file='project_video.mp4',
                    out_file='./out.mp4',
                    calibration_files= 'camera_cal/calibration*.jpg')
