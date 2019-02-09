# Line_Detection

This project was mainly made for an airship prototype in which the objective was to give a PID control direction vectors. One giving the direction to the closest point from the camera center to the line and the second one being the line’s direction. The program gives visual simulation of what the camera sees, and it will take the biggest norm vector of the line object detected with the canny mask. This project is done on Python3 and with the use of OpenCV. 

Video_Line_Detection.py is the main program it detects the largest single line of the color that you wish to detect. It also gives shows a video simulation as if the airship was moving, there is a red point that determines the camera center and the lines of the object detected.

Camera_Data has my own camera calibration Matrix and Distortion, in the case of the video since the video camera is already calibrated it isn’t used so the distortion vector and the camera matrix whose values are zero and an identity matrix.

Functions contain de various functions used to calculate the vectors, the undistorted image (if needed) and the center of the camera shift.
