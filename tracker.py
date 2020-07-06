import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

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

nonzero = []
for i in range((len(images) - 1)):
    mask = cv2.absdiff(images[i], images[i + 1])
    # mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                              cv2.THRESH_BINARY, 11, 2)
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plot_img = normal_images[i]
    hierarchy = hierarchy[0]
    for c in cnts:
        c = cv2.convexHull(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if 8 < len(approx) < 15 and 512 < area < 952 and 83 < perimeter < 109:
            print("Area =  {} | Points = {} | Perimeter = {}".format(area, len(approx), perimeter))

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.drawContours(plot_img, [c], -1, (0, 255, 255), 1)

            cv2.putText(plot_img, str(len(approx)) + "|" + str(area), (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # except Exception as e:
        #     pass

    print("=========================================================================================")
    # cv2.drawContours(images[i], contours, -1, (255, 255, 0), 1)
    cv2.imshow('Image', plot_img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
