import numpy as np
import cv2
import sys
import time

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients):
    axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, matrix_coefficients, distortion_coefficients)
    
    # Ensure tvec is a 2D point
    origin = (int(tvec[0]), int(tvec[1]))
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    print("Origin:", origin)
    print("Image Points:", imgpts)
    
    if imgpts.shape[0] != 3:
        print("Error: imgpts does not contain 3 points. Current shape:", imgpts.shape)
        return
    
    cv2.line(frame, origin, tuple(imgpts[0].ravel()), (255, 0, 0), 5)  # X axis
    cv2.line(frame, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Y axis
    cv2.line(frame, origin, tuple(imgpts[2].ravel()), (0, 0, 255), 5)  # Z axis

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            # Ensure rvec and tvec are in the correct shape
            rvec = rvec.reshape((3, 1))
            tvec = tvec.reshape((3, 1))
            
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients)
    
    return frame

aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()

intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
distortion = np.array((-0.43948, 0.18514, 0, 0))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
    
    cv2.imshow('Estimated Pose', output)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()