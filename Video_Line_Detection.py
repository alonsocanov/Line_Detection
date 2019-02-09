import functions as f
import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt


def main():
    print('Line detection')

    # lower bound of color line
    low_color = [0, 0, 0]
    # upper bound of color line
    up_color = [10, 10, 10]
    # lower and upper color normalization for HSV on OpenCV
    low_color, up_color = f.hsv2cvhsc(low_color, up_color)

    # radious for visualization image 
    rad = 10
    # path of the test videos
    path = 'Airship_Videos/'
    # video test name
    vid = 'black_line.mp4'
    # vid = 'black_line_out.mp4'
    video = cv2.VideoCapture(path + vid)
    line = None

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
    # create text for figures
    corner_text = (20, 20)
    corner_text_height = (20, 50)
    corner_text_accel = (20, 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2




    while True:
        ret, orig_frame = video.read()
        if not ret:
            video = cv2.VideoCapture(path + '/' + vid)
            continue
        calib_frame, center_image, img_dim = f.cam_calib(orig_frame, cam_matrix, cam_disto, cam_scaled_matrix, image_type)
        frame = cv2.GaussianBlur(calib_frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(low_color)
        upper = np.array(up_color)
        mask = cv2.inRange(hsv, lower, upper)
        edges = cv2.Canny(mask, 50, 150)
        threshold = 50
        # lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, maxLineGap=50)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=5, maxLineGap=10) 
       
       
        # setting variables to zero, because there isn't a file to read 
        gyro = np.array([0.,0.,0.])
        height = 0.0


        center_image = f.center_camera(gyro, height, center_image) 
        cx, cy = center_image.astype(int)
        max_line = np.array([0, 0, 0, 0])
        if lines is not None:
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
        coor, direction, norm_pixel, norm_wu = f.dis(line, center_image, cam_matrix, height)
        x1, y1, x2, y2 = line.astype(int)
            

        # Camera center
        cv2.circle(frame, (cx, cy), rad, (0, 0, 225), -1)
        # Lines for visualisation
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.line(frame, (cx, cy), (coor[0], coor[1]), (255, 0, 0))
        cv2.line(frame, (coor[0], coor[1]), (direction[0], direction[1]), (255, 0, 0))

        # image visualization
        cv2.imshow(img_frame, frame)
        cv2.putText(edges, 'Norm: '+ str(norm_wu), corner_text, font, font_scale, font_color, line_type)
        cv2.putText(edges, 'Height: '+ str(height), corner_text_height, font, font_scale, font_color, line_type)
        str_gyro = str(gyro[0]) + ' ' + str(gyro[1]) + ' ' + str(gyro[2]) 
        cv2.putText(edges, 'gyro: '+ str_gyro, corner_text_accel, font, font_scale, font_color, line_type)
        cv2.imshow(img_edges, edges)

        # if q is pressed the vido stops
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

