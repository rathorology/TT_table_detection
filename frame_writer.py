import cv2


video = 'set2-1.MOV'

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video)
cnt = 0

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, first_frame = cap.read()

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        # removing scorecard
        roi = frame

        cv2.imshow("image", roi)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        cv2.imwrite('frames/' + str(cnt) + '.png', roi)
        cnt = cnt + 1

    # Break the loop
    else:
        break

cv2.destroyAllWindows()
