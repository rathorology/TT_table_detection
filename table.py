import cv2
import numpy as np

cap = cv2.VideoCapture("set2-1.MOV")

while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color Range for table
    lower_blue = np.array([94, 51, 170])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cordinates = list()
    for c in contours:
        c = cv2.convexHull(c)
        area = cv2.contourArea(c)

        perimeter = cv2.arcLength(c, True)
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            if area > (frame.shape[1] * frame.shape[0]) / 440:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the contour and center of the shape on the image
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
                cv2.putText(frame, "Table", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                for p in approx.tolist():
                    cordinates.append((p[0][0], p[0][1]))

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
