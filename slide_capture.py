import numpy as np
import cv2

# Capture video from camera
cap = cv2.VideoCapture(0)

# or playing video from file
# cap = cv2.VideoCapture('...')

while(True): # cap.isOpened()
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()