import numpy as np
import cv2


def getFaceCoordinates(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces -- [grayscale image, scaleFactor, minNeighbours]
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    faces_coordinates = face_cascade.detectMultiScale(gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))

    return faces_coordinates

def testGetFaceCoordinates():
    # Read the input image
    img = cv2.imread('Photos_Directory/students.jpg')
    face_coordinates = getFaceCoordinates(img)

    # Draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

    

testGetFaceCoordinates()