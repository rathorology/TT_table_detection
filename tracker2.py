import cv2
import numpy as np

# cap = cv2.VideoCapture('/home/rathorology/PycharmProjects/tracking/6ft.mp4')
cap = cv2.VideoCapture('set2-1.MOV')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # contour = cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if 280 < cv2.contourArea(contour) < 1500 and 7 < len(approx) < 15:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 3)
            cv2.putText(frame1, str(len(approx)) + "|" + str(area) + "|" + str(round(perimeter, 1)), (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    # image = cv2.resize(frame1, (1280, 720))
    # out.write(image)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Dilated", dilated)
    cv2.imshow("Frame", frame1)
    cv2.waitKey(0)
    frame1 = frame2
    ret, frame2 = cap.read()

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
out.release()
