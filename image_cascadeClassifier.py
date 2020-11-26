import numpy as np
import cv2
from scipy import spatial

def get_center_coordinate(coordinate_list):
    center_coordinate_list = []
    for eye in coordinate_list:
        (x, y, w, h) = eye
        center_coordinate = (x + 0.5 * w, y + 0.5 * h)
        center_coordinate_list.append(center_coordinate)
    return center_coordinate_list

def get_center_and_radius(eye_center_list):
    coordinate_list=[]
    while len(eye_center_list) > 1:
        center=eye_center_list.pop(0)
        tree = spatial.KDTree(eye_center_list)

        distance, center_pair_index=tree.query([center])
        distance=distance[0]
        (center_pair_x, center_pair_y)=eye_center_list[center_pair_index[0]]

        (center_x, center_y)=center

        center_face_x = (center_x + center_pair_x) / 2
        center_face_y = (center_y + center_pair_y) / 2

        eye_center_list.pop(center_pair_index[0])

        x_radius = distance * 3.5
        y_radius = distance * 4.5
        face_coordinate = (int(center_face_x - x_radius / 2), int(center_face_y - y_radius / 2),
                           int(x_radius), int(y_radius))
        coordinate_list.append(face_coordinate)
    return coordinate_list


def get_faces_by_eyes(eyes_list):
    eyes_center_list=get_center_coordinate(eyes_list)
    return get_center_and_radius(eyes_center_list)

def getFaceCoordinates(img, faceCascade):
    # Load the cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )
    faces = get_faces_by_eyes(eyes)


    return faces


# -----------------------SPLIT SYSTEM------------------------
def getListOfFaces(img, face_locations):
    width = height = 224
    list_of_faces = []
    for face_coords in face_locations:
        (x, y, w, h) = face_coords
        x=max(0, x)
        y=max(0, y)
        right=min(len(img[0]), x+w)
        top=min(len(img), y+h)
        face_image = img[y:top,x:right]
        resized = cv2.resize(face_image, (width, height))

        list_of_faces.append(resized)

    return list_of_faces


def testGetFaceCoordinates():
    # Read the input image
    img = cv2.imread('Photos_Directory/students.jpg')
    face_finder = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    face_coordinates = getFaceCoordinates(img, face_finder)
    # Draw rectangle around the faces\
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

if __name__ == '__main__':
    testGetFaceCoordinates()
