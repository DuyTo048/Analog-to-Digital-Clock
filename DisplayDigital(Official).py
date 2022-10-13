import cv2
import math
import numpy as np
import tkinter as tk

from matplotlib import pyplot as plt
from math       import sqrt, acos, degrees


# Reading the input image and convert the original RGB to a grayscale image
kernel   = np.ones((5, 5), np.uint8)
img1     = cv2.imread('input1.jpg')
img      = cv2.imread('input1.jpg', 0)
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# Appling a binary threshold to the image
ret, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)


# Create mask
height, width = img.shape

mask  = np.zeros((height, width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)


# Circle Detection
cimg    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.2, 100)

for i in circles[0,:]:
    i[2] = i[2] + 4
    # cv2.cicle(image, center_coordinates, radius, color, thickness)
    cv2.circle(mask, (int(i[0]),int(i[1])), int(i[2]), (255,255,255), thickness = -1)


# Copy that image using that mask
masked_data = cv2.bitwise_and(img1, img1, mask = mask)


# Apply threshold
_,thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)


# Find Contour
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h          = cv2.boundingRect(contours[0])


# Crop masked_data
crop = masked_data[y + 30 : y + h -30, x + 30 : x + w - 30]

height, width, channel = crop.shape
blur_crop     = cv2.GaussianBlur(crop, (5, 5), 0)
edges         = cv2.Canny(blur_crop, 50, 150)


# Line segments
line_image = np.copy(crop) * 0
lines      = cv2.HoughLinesP(edges, 1, np.pi/180, 15, np.array([]), 100, 10)

l = []

xl1, xl2, yl1, yl2 = 0, 0, 0, 0             #long   -> l
xm1, xm2, ym1, ym2 = 0, 0, 0, 0             #medium -> m
xs1, xs2, ys1, ys2 = 0, 0, 0, 0             #short  -> s

hypo = 0
# Getting the values from the line
for line in lines:
    
    x1, y1, x2, y2 = line[0]
    
    dx = x2 - x1
    if (dx < 0):
        dx = dx* (-1)
        
    dy = y2 - y1
    if (dy < 0):
        dy = dy* (-1)
        
    hypo = sqrt(dx**2 + dy**2)  
    l.append(hypo)

l.sort(reverse=True)

s, m, h = 0, 0, 0

for f in range(9):
    
    for line in lines:
        # getting the values from the line
        x1, y1, x2, y2 = line[0]
        
        #cv2.line(crop, (x1, y1), (x2, y2), (0, 255, 0), 3)
        dx = x2 - x1
        if (dx < 0):
            dx = dx* (-1)
            
        dy = y2 - y1
        if (dy < 0):
            dy = dy* (-1)
        
        hypo = sqrt(dx**2 + dy**2)

        if (hypo == l[0]):
            m = hypo
            xl1 = x1
            xl2 = x2
            yl1 = y1
            yl2 = y2

            # getting line region
            cv2.line(crop, (xl1, yl1), (xl2, yl2), (255, 0, 0), 3)

        if (m == l[0]):
            if (hypo == l[f]):
                if ((sqrt((xl2 - x2)**2 + (yl2 - y2)**2)) > 20):
                    if ((sqrt((xl1 - x1)**2 + (yl1 - y1)**2)) > 20):
                        xs1 = x1
                        xs2 = x2
                        ys1 = y1
                        ys2 = y2

                        # getting line region
                        cv2.line(crop, (xl1, yl1), (xl2, yl2), (0, 255, 0), 5)
                        h = 1
                        break
                    
# Calculate center point
xcenter = width/2
ycenter = height/2


# Determine the cooridnates of the end point (farther from the center)
def coordinates (x1, y1, x2, y2):
    a = abs(xcenter - x1)
    b = abs(xcenter - x2)

    if (a > b):
        x_coor = x1
        y_coor = y1
    else:
        x_coor = x2
        y_coor = y2
        
    return x_coor, y_coor

xhour, yhour = coordinates(xs1, ys1, xs2, ys2)
xmin, ymin   = coordinates(xl1, yl1, xl2, yl2)
xsec, ysec   = coordinates(xm1, ym1, xm2, ym2)

cv2.line(crop, (xs1, ys1), (xs2, ys2), (0, 255, 0), 5)


# Calculate the Hour, Minute, Second-hands by the law of cosines
def law_of_cosines (x, y):
    l1 = sqrt(((xcenter - x)**2) + ((ycenter - y)**2))
    l2 = ycenter
    l3 = sqrt(((xcenter - x)**2) + ((0 - y)**2))
    
    cos_theta = ( (l1**2) + (l2**2) - (l3**2) )/(2*l1*l2)
    theta_radian = acos(cos_theta)
    theta = math.degrees(theta_radian)
    return theta

theta_hour = law_of_cosines(xhour, yhour)
theta_min  = law_of_cosines(xmin, ymin)
theta_sec  = law_of_cosines(xsec, ysec)

# Calculate the the time of each hands
def time_cal (x, y):
    if (y > xcenter):
        right = 1
    else:
        right = 0 
        
    if (y == xhour):
        if (right == 1):
            a = int(x/30)
        else:
            a = 12 - int(x/30)
        if a == 0:
            a = 12
    else:
        if (right == 1):
            a = int(x/6)
        else:
            a = 60 - int(x/6)
            if (y == xcenter):
                a = 30
    return a

hour   = time_cal(theta_hour, xhour)
minute = time_cal(theta_min, xmin)
sec    = time_cal(theta_sec, xsec)


# Display window
canvas = tk.Tk()
canvas.title("Analog to Digital")
canvas.geometry("500x250")

digit = tk.Label(canvas, font = ("ds-digital", 65, "bold"), bg = "white", fg = "blue", bd = 80)
digit.grid(row = 0, column = 1)


# Display result
def display(hour, minute, sec):
    value = "{0:0=2d}:{1:0=2d}:{2:0=2d}".format(hour, minute, sec)
    digit.config(text=value)
    print(value)

display(hour, minute, sec)  
canvas.mainloop()

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 1)

lines_edges = cv2.addWeighted(crop, 0.8, line_image, 1, 0)
cv2.imshow('Line Image', line_image)
cv2.imshow('Crop', crop)
cv2.waitKey(0)

