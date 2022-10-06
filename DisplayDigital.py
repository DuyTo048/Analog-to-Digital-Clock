import cv2
import math
import numpy as np
import tkinter as tk

from matplotlib import pyplot as plt
from math       import sqrt, acos, degrees


# Reading the input image and convert the original RGB to a grayscale image
kernel = np.ones((5, 5), np.uint8)
img1   = cv2.imread('input1.jpg')
img    = cv2.imread('input1.jpg', 0)
img_gray   = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# Appling a binary threshold to the image
ret, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)


# Create mask
height, width = img.shape

mask  = np.zeros((height, width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)


# cv2.imshow('detected', gray)
cimg    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.2, 100)

for i in circles[0,:]:
    i[2] = i[2] + 4
    # Draw on mask
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

kernel_size    = 5
blur_crop      = cv2.GaussianBlur(crop, (kernel_size, kernel_size), 0)
low_threshold  = 50
high_threshold = 150
edges          = cv2.Canny(blur_crop, low_threshold, high_threshold)

rho             = 1                     # distance resolution in pixels
theta           = np.pi / 180           # angular resolution in radians
threshold       = 15                    # minimum number of votes
min_line_length = 100                   # minimum number of pixels making up a line
max_line_gap    = 10                    # maximum gap in pixels between connectable


## Line segments
line_image      = np.copy(crop) * 0


# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

l = []

xl1, xl2, yl1, yl2 = 0, 0, 0, 0             #long   -> l
xm1, xm2, ym1, ym2 = 0, 0, 0, 0             #medium -> m
xs1, xs2, ys1, ys2 = 0, 0, 0, 0             #short  -> s

for line in lines:

    # getting the values from the line
    x1, y1, x2, y2 = line[0]
    dx = x2 - x1

    if (dx < 0):
        dx = dx * -1
    dy = y2 - y1

    if (dy < 0):
        dy = dy * -1
    hypo = sqrt(dx ** 2 + dy ** 2)

    l.append(hypo)

a = len(l)  # -> 295
l.sort(reverse=True)

sec, minute, hour = 0, 0, 0


for f in range(a):

    for line in lines:

        # getting the values from the line
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1

        if (dx < 0):
            dx = dx * -1
        dy = y2 - y1

        if (dy < 0):
            dy = dy * -1
        hypo2 = sqrt(dx ** 2 + dy ** 2)

        if (hypo2 == l[0]):
            minute = hypo2
            xl1 = x1
            xl2 = x2
            yl1 = y1
            yl2 = y2

            # getting line region
            cv2.line(crop, (xl1, yl1), (xl2, yl2), (255, 0, 0), 3)

        if (minute == l[0]):

            if (hypo2 == l[f]):

                if ((sqrt((xl2 - x2) ** 2 + (yl2 - y2) ** 2)) > 20):

                    if ((sqrt((xl1 - x1) ** 2 + (yl1 - y1) ** 2)) > 20):
                        xm1 = x1
                        xm2 = x2
                        ym1 = y1
                        ym2 = y2

                        # getting line region
                        cv2.line(crop, (xs1, ys1), (xs2, ys2), (0, 255, 0), 3)
                        hour = 1
                        break

    if (hour == 1):
        break

xcenter = int(width / 2)
ycenter = int(height / 2)

hour1 = abs(xcenter - xs1)
hour2 = abs(xcenter - xs2)

if (hour1 > hour2):
    xhour = xs1
    yhour = ys1
else:
    xhour = xs2
    yhour = ys2

min1 = abs(xcenter - xl1)
min2 = abs(xcenter - xl2)

if (min1 > min2):
    xmin = xl1
    ymin = yl1
else:
    xmin = xl2
    ymin = yl2

l1 = sqrt(((xcenter - xhour) ** 2) + ((ycenter - yhour) ** 2))
l2 = ycenter
l3 = sqrt(((xcenter - xhour) ** 2) + ((0 - yhour) ** 2))

cos_theta_hour = (((l1) ** 2) + ((l2) ** 2) - ((l3) ** 2)) / (2 * (l1) * (l2))
theta_hours_radian = acos(cos_theta_hour)
theta_hours = math.degrees(theta_hours_radian)

if (xhour > xcenter):
    right = 1
else:
    right = 0

if (right == 1):
    hour = int(theta_hours / (6 * 5))
else:
    hour = 12 - (int(theta_hours / (6 * 5)))

if (hour == 0):
    hour = 12

l1 = sqrt(((xcenter - xmin) ** 2) + ((ycenter - ymin) ** 2))
l2 = ycenter
l3 = sqrt(((xcenter - xmin) ** 2) + ((0 - ymin) ** 2))

cos_theta_min = (((l1) ** 2) + ((l2) ** 2) - ((l3) ** 2)) / (2 * (l1) * (l2))
theta_min_radian = acos(cos_theta_min)
theta_min = math.degrees(theta_min_radian)

if (xmin > xcenter):
    right = 1
else:
    right = 0

if (right == 1):
    minute = int(theta_min / ((6 * 5) / 5))
elif (right == 0):
    minute = 60 - (int(theta_min / ((6 * 5) / 5)))
    if (xmin == xcenter):
        minutes = 30

if (minute < 10):
    def display():
        value = "{}:0{}".format(hour, minute)
        digit.config(text=value)
else:
    def display():
        value = "{}:{}".format(hour, minute)
        digit.config(text=value)


## Display window
canvas = tk.Tk()
canvas.title("Analog to Digital")
canvas.geometry("400x250")

digit = tk.Label(canvas, font = ("ds-digital", 65, "bold"), bg = "white", fg = "blue", bd = 80)
digit.grid(row = 0, column = 1)


## Display result
display()
canvas.mainloop()
