import numpy as np
import cv2
import sys
import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from objloader import OBJ

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

def init_gl(width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def draw_cube(size):
    glBegin(GL_QUADS)
    glVertex3f( size,  size, -size)
    glVertex3f(-size,  size, -size)
    glVertex3f(-size,  size,  size)
    glVertex3f( size,  size,  size)
    
    glVertex3f( size, -size,  size)
    glVertex3f(-size, -size,  size)
    glVertex3f(-size, -size, -size)
    glVertex3f( size, -size, -size)
    
    glVertex3f( size,  size,  size)
    glVertex3f(-size,  size,  size)
    glVertex3f(-size, -size,  size)
    glVertex3f( size, -size,  size)
    
    glVertex3f( size, -size, -size)
    glVertex3f(-size, -size, -size)
    glVertex3f(-size,  size, -size)
    glVertex3f( size,  size, -size)
    
    glVertex3f(-size,  size,  size)
    glVertex3f(-size,  size, -size)
    glVertex3f(-size, -size, -size)
    glVertex3f(-size, -size,  size)
    
    glVertex3f( size,  size, -size)
    glVertex3f( size,  size,  size)
    glVertex3f( size, -size,  size)
    glVertex3f( size, -size, -size)
    glEnd()

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
            
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Combine rotation and translation into a single 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.reshape(3)
            
            # Set up OpenGL modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glMultMatrixf(transformation_matrix.T)
            
            # Draw the 3D cube
            glColor3f(0.0, 1.0, 0.0)  # Set color to green
            draw_cube(0.02)  # Draw a cube with side length 0.02 (same as marker size)

    return frame

aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()

intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
distortion = np.array((-0.43948, 0.18514, 0, 0))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize Pygame and OpenGL
pygame.init()
display = (1280, 720)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
init_gl(1280, 720)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Clear the OpenGL buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Set up the camera matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    
    # Process the frame and estimate pose
    output = pose_estimation(frame, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
    
    # Convert the OpenCV output to a Pygame surface
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = np.rot90(output)
    output = pygame.surfarray.make_surface(output)
    
    # Display the Pygame surface
    screen = pygame.display.get_surface()
    screen.blit(output, (0,0))
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

cap.release()
cv2.destroyAllWindows()