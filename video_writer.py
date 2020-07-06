import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

import cv2
import numpy as np
import imutils

video = 'set2-1.MOV'

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video)
cnt = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# reading frames
images = []
normal_images = []
for i in frames:
    img = cv2.imread('frames/' + i)
    normal_images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (25, 25), 0)
    images.append(img)

images = np.array(images)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height), False)
nonzero = []
cnt = 0
for i in range((len(images) - 1)):
    mask = cv2.absdiff(images[i], images[i + 1])
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)
    # cv2.imwrite('new_frames/' + str(cnt) + '.png', mask)
    cnt = cnt + 1
    out.write(mask)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
