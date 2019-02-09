import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt

# closest distnce to a point in the line
def dis(lines, center_image, cam_matrix, height):
    center_image = np.array(center_image)
    center_image = center_image.astype(float)
    # if point a_y is smaller than b_y invert values of a and b to have same triangle
    if lines[1] < lines[3]:
        b = np.array([lines[0], lines[1]])
        a = np.array([lines[2], lines[3]])
    else:
        a = np.array([lines[0], lines[1]])
        b = np.array([lines[2], lines[3]])
    # norms of the thre sides of the triangle 
    norm_a = cv2.norm(center_image, a, cv2.NORM_L2)
    norm_b = cv2.norm(b, center_image, cv2.NORM_L2)
    norm_base = cv2.norm(b, a, cv2.NORM_L2)
    # direction of the line object found (slope)
    slope = b - a
    # pixel coordinate of the closest point to the center image
    lenght = ((slope[0]*(center_image[0] - a[0])) + (slope[1]*(center_image[1] - a[1]))) / (slope[0]**2 + slope[1]**2)
    coor_pix = np.array(a + slope * lenght)
    norm_cc = cv2.norm(coor_pix, center_image, cv2.NORM_L2)
    if coor_pix[1] > a[1]:
        coor_pix = a
    elif coor_pix[1] < b[1]:
        coor_pix = b
    #same line vector
    norm_p = cv2.norm(b, a, cv2.NORM_L2)
    if norm_p == 0:
        norm_p = .000001
    unit_line_dir = slope / norm_p
    direction_ = unit_line_dir*norm_p + coor_pix

    # world units
    focal_lenght = np.array([cam_matrix[0,0], cam_matrix[1,1]])
    wu_coor_pix = coor_pix*(height)/focal_lenght
    wu_ci_ = center_image*(height)/focal_lenght
    norm_wu = cv2.norm(wu_coor_pix, wu_ci_, cv2.NORM_L2)
    return coor_pix.astype(int), direction_.astype(int), norm_cc, norm_wu

    # calculate height of the airship
def center_camera(gyro, height, center_image):
    shift =  height * np.tan(gyro)
    pos = center_image + shift[:2]
    return pos

# camera Calibration givin the calibrated image
def cam_calib(orig_frame, cam_matrix, cam_disto, cam_scaled_matrix, undisort):
    # Undistort an image
    if undisort:
        balance = 1
        DIM = orig_frame.shape[:2]
        dim1 = orig_frame.shape[:2]
        dim2 = (int(dim1[0]/1.1), int(dim1[1]/1.1))
        dim3 = (int(dim1[0]/1), int(dim1[1]/1))

        assert dim1[0]/dim1[1] == DIM[0]/DIM[1] 
        if dim2 == None:
            dim2 = dim1
        if dim3 == None:
            dim3 = dim1

        mapx,mapy = cv2.fisheye.initUndistortRectifyMap(cam_scaled_matrix,cam_disto,np.eye(3),cam_matrix,dim3,cv2.CV_16SC2)

        dst = cv2.remap(orig_frame,mapx,mapy,interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        center_image = np.array([dst.shape[0]/2, dst.shape[1]/2])
        return dst, center_image, dst.shape[0:2]
    else:
        center_image = np.array([orig_frame.shape[0]/2, orig_frame.shape[1]/2])
        return orig_frame, center_image, orig_frame.shape[:2]
