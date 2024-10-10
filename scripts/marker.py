import cv2
import os
import numpy as np
from cv2 import aruco
import math
import matplotlib.pyplot as plt

image = cv2.imread("Path to image with ArUco marker")
image2 = cv2.imread("Path to the poster")


def image_shower(image):
    cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


height, width, chhanel = image2.shape
topleft = [0, 0]
topright = [width, 0]
bottomright = [width, height]
bottomleft = [0, height] #left bottom

c = np.array([topleft, topright, bottomright, bottomleft], dtype=np.float32)
[x, y] = center = np.mean(c, axis=0)

w = width / 6
h = height / 6
corner1 = (x - w / 2, y - h / 2)
corner2 = (x + w / 2, y - h / 2)
corner3 = (x + w / 2, y + h / 2)
corner4 = (x - w / 2, y + h / 2)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
arucocorners, ids, Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

Arucoc1 = arucocorners[0][0][0]
Arucoc2 = arucocorners[0][0][1]
Arucoc3 = arucocorners[0][0][2]
Arucoc4 = arucocorners[0][0][3]



input_corners = np.float32([corner1,corner2,corner3,corner4])
output_corners = np.float32([Arucoc1,Arucoc2,Arucoc3,Arucoc4])

M = cv2.getPerspectiveTransform(input_corners, output_corners)


input_corners = np.float32([topleft,topright,bottomright,bottomleft])
final_corners = cv2.perspectiveTransform(input_corners.reshape(-1, 1, 2), M)


sac1=final_corners[0,0]
sac2=final_corners[1,0]
sac3=final_corners[2,0]
sac4=final_corners[3,0]


transformed_image2= cv2.warpPerspective(image2, M,(image.shape[1], image.shape[0]))
image_shower(transformed_image2)
pts = np.array([sac1,sac2,sac3,sac4], np.int32)

# Create mask with zeros
mask = np.zeros_like(image)
cv2.fillPoly(mask, [pts], (255, 255, 255))

# Invert mask to keep everything outside the polygon area
mask = cv2.bitwise_not(mask)
image_shower(mask)

# Apply mask to image to remove the polygon area
image = cv2.bitwise_and(image, mask)
image_shower(image)
final_image = cv2.bitwise_or(image, transformed_image2)
image_shower(final_image)
cv2.imwrite("Output image path",final_image)