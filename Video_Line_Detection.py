import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt

print('Line detection for airship project')


# lower bound of color line
low_color = [0, 0, 0]
# upper bound of color line
up_color = [10, 10, 10]
# radious for visualization image 
rad = 10
# path of the test videos
path = 'Airship_Videos/'
# video test name
vid = 'black_line.mp4'
# vid = 'black_line_out.mp4'
# video capture
video = cv2.VideoCapture(path + vid)
line = None

 # load data from arduino
path_file_arduino = 'Arduino/'
arduino_data = 'arduino_data_v1.txt'
# load camera parameters
# image_type: camera aleady calibrated = 0, normal camera = 1, camera fiseye = 2
image_type = 0
path_cam_data = 'Camera_Data/'
# path for camera matrix parameters
if image_type == 1:
    version = '720x480'
    # path for camera matrix parameters
    file_cam_matrix = 'Camera_Matrix_v'+version+'.txt'
    # path for camera matrix scaled parameters
    file_cam_scaled_matrix = 'Camera_Scaled_Matrix'+'.txt'
    # path for camera distortion parameters
    file_cam_distorion = 'Camera_Distortion_v'+version+'.txt'
elif image_type == 2:
    version = '640x480'
    #version = '752x480'
    # path for camera matrix parameters
    file_cam_matrix = 'Camera_new_Matrix_FE_v'+version+'.txt'
    # path for camera matrix scaled parameters
    file_cam_scaled_matrix = 'Camera_Scaled_Matrix_FE_v'+version+'.txt'
    # path for camera distortion parameters
    file_cam_distorion = 'Camera_Distortion_FE_v'+version+'.txt'
else:
    # path for camera matrix parameters
    file_cam_matrix = 'Camera_Matrix'+'.txt'
    # path for camera matrix scaled parameters
    file_cam_scaled_matrix = 'Camera_Scaled_Matrix'+'.txt'
    # path for camera distortion parameters
    file_cam_distorion = 'Camera_Distortion'+'.txt'
    # load text files as float
cam_matrix = np.loadtxt(path_cam_data + file_cam_matrix, delimiter=',')
cam_disto = np.loadtxt(path_cam_data + file_cam_distorion, delimiter=',')
cam_scaled_matrix = np.loadtxt(path_cam_data + file_cam_scaled_matrix, delimiter=',')
# Print camera parameters
print('Camera Parameters')
print('Camera Matrix')
print(cam_matrix)
print('Camera Distortion')
print(cam_disto)

# Create figures
img_frame = 'Frame'
cv2.namedWindow(img_frame)
cv2.moveWindow(img_frame, 40, 30)
img_edges = 'Edges'
cv2.namedWindow(img_edges)
cv2.moveWindow(img_edges, 450, 30)
# create text
corner_text = (20, 20)
corner_text_height = (20, 50)
corner_text_accel = (20, 80)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
line_type = 2

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
    lenght = ((dir_[0]*(center_image[0] - a[0])) + (dir_[1]* (center_image_[1] - a[1]))) / (dir_[0]**2 + dir_[1]**2)
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

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture(path + '/' + vid)
        continue
    calib_frame, center_image, img_dim = cam_calib(orig_frame, cam_matrix, cam_disto, cam_scaled_matrix, image_type)
    frame = cv2.GaussianBlur(calib_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(low_color)
    upper = np.array(up_color)
    mask = cv2.inRange(hsv, lower, upper)
    edges = cv2.Canny(mask, 50, 150)
    threshold = 50
    # lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, maxLineGap=50)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=5, maxLineGap=10) 
   
   
    
    gyro = np.array([0.,0.,0.])
    height = 0.0


    center_image = center_camera(gyro, height, center_image) 
    cx, cy = center_image.astype(int)
    max_line = np.array([0, 0, 0, 0])
    if lines is not None:
        #if np.all(lines.shape, (4,), 0):
         #   line = lines
        if True:
            for l in lines:
                a = max_line[:2]
                b = max_line[2:4]
                c = l[0,:2]
                d = l[0, 2:4]
                n1 = cv2.norm(a, b, cv2.NORM_L2)
                n2 = cv2.norm(c, d, cv2.NORM_L2)
                if n1 < n2:
                    max_line = l[0,:4]
            line = max_line.astype(float)
    else:
        if line.shape[0] != 4 :
            lim = np.array([30, 30, 30, -30])
            line = np.concatenate((center_image, center_image),0)+lim
            line = line.astype(float)
    coor, direction, norm_pixel, norm_wu = dis(line, center_image, cam_matrix, height)
    x1, y1, x2, y2 = line.astype(int)
        

    # Camera center
    cv2.circle(frame, (cx, cy), rad, (0, 0, 225), -1)
    # Lines for visualisation
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.line(frame, (cx, cy), (coor[0], coor[1]), (255, 0, 0))
    cv2.line(frame, (coor[0], coor[1]), (direction[0], direction[1]), (255, 0, 0))

    
    cv2.imshow(img_frame, frame)
    cv2.putText(edges, 'Norm: '+ str(norm_wu), corner_text, font, font_scale, font_color, line_type)
    cv2.putText(edges, 'Height: '+ str(height), corner_text_height, font, font_scale, font_color, line_type)
    str_accel = str(gyro[0]) + ' ' + str(gyro[1]) + ' ' + str(gyro[2]) 
    cv2.putText(edges, 'gyro: '+ str_accel, corner_text_accel, font, font_scale, font_color, line_type)
    cv2.imshow(img_edges, edges)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
