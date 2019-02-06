import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt

# closest distnce to a point in the line
def dis(lines_, center_image_, cam_matrix_, height_):
    center_image_ = np.array(center_image_)
    center_image_ = center_image_.astype(float)
    # if point a_y is smaller than b_y invert values of a and b to have same triangle
    if lines_[1] < lines_[3]:
        b = np.array([lines_[0], lines_[1]])
        a = np.array([lines_[2], lines_[3]])
    else:
        a = np.array([lines_[0], lines_[1]])
        b = np.array([lines_[2], lines_[3]])

    norm_a = cv2.norm(center_image_, a, cv2.NORM_L2)
    norm_b = cv2.norm(b, center_image_, cv2.NORM_L2)
    norm_base = cv2.norm(b, a, cv2.NORM_L2)

    dir_ = b - a
    lenght = ((dir_[0]*(center_image_[0] - a[0])) + (dir_[1]* (center_image_[1] - a[1]))) / (dir_[0]**2 + dir_[1]**2)
    coor_ = np.array(a + dir_ * lenght)
    norm_cc = cv2.norm(coor_, center_image_, cv2.NORM_L2)
    if coor_[1] > a[1]:
        coor_ = a
    elif coor_[1] < b[1]:
        coor_ = b
    #same line vector
    slope_ = b - coor_
    norm_p = cv2.norm(b, coor_, cv2.NORM_L2)
    if norm_p == 0:
        norm_p = .000001
    unit_line_dir = slope_ / norm_p
    direction_ = unit_line_dir*norm_p + coor_

    # world units
    focal_lenght_ = np.array([cam_matrix_[0,0], cam_matrix_[1,1]])
    wu_coor_ = coor_*(height_)/focal_lenght_
    wu_ci_ = center_image_*(height_)/focal_lenght_
    norm_wu = cv2.norm(wu_coor_, wu_ci_, cv2.NORM_L2)
    return coor_.astype(int), direction_.astype(int), norm_cc, norm_wu

    # calculate height of the airship
def center_camera(accel_, height_, center_image_):
    shift_ =  height_ * np.tan(accel_)
    pos_ = center_image_ + shift_[:2]
    return pos_

# camera Calibration givin the calibrated image
def cam_calib(orig_frame_, cam_matrix_, cam_disto_, cam_scaled_matrix_, undisort_):
    # undisort
    # Undistort an image
    if undisort_:
        balance = 1
        DIM = orig_frame_.shape[:2]
        dim1 = orig_frame_.shape[:2]
        dim2 = (int(dim1[0]/1.1), int(dim1[1]/1.1))
        dim3 = (int(dim1[0]/1), int(dim1[1]/1))

        assert dim1[0]/dim1[1] == DIM[0]/DIM[1] 
        if dim2 == None:
            dim2 = dim1
        if dim3 == None:
            dim3 = dim1

        mapx,mapy = cv2.fisheye.initUndistortRectifyMap(cam_scaled_matrix_,cam_disto_,np.eye(3),cam_matrix_,dim3,cv2.CV_16SC2)

        dst = cv2.remap(orig_frame_,mapx,mapy,interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        center_image_ = np.array([dst.shape[0]/2, dst.shape[1]/2])
        return dst, center_image_, dst.shape[0:2]
    else:
        center_image_ = np.array([orig_frame_.shape[0]/2, orig_frame_.shape[1]/2])
        return orig_frame_, center_image_, orig_frame_.shape[:2]
