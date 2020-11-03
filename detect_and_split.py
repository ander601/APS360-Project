import face_recognition
import cv2
import os


#-------------------FACE DETECTION SYSTEM-------------------
#returns a list of coordinates; 1 set for each face in the photo. 
def getCoordinates(img):
    return face_recognition.face_locations(img)

#-----------------------SPLIT SYSTEM------------------------
def getListOfFaces(img):
    face_locations = getCoordinates(img)

    width = height = 244
    list_of_faces = []
    
    for face_coords in face_locations:
        (top, right, bottom, left) = face_coords
        
        face_image = img[top:bottom, left:right]
        resized = cv2.resize(face_image, (width, height))
        
        list_of_faces.append(resized)

    return list_of_faces


###################### Notes on Reading Coordinates #######################

# face_locations is an array listing the co-ordinates of each face.
# for example, an image containg 5 faces would return 5 sets of coordinates
# > face_locations = face_recognition.face_locations(img)
# > print (face_locations)
# >>>  [(126, 394, 216, 305), (146, 185, 236, 96), (56, 484, 146, 394), (176, 534, 265, 444), (156, 285, 245, 195)]

#----------reading coordinates----------
# ex: (146, 185, 236, 96)
# x = 185, y = 236 is bottom right corner
# x = 185, y = 146 is top right corner
# x = 96,  y = 146 is top left corner
# x = 96,  y = 236 is bottom left corner

###################### For testing & demonstration #######################

def testFaceDetection(img):
    face_locations = getCoordinates(img)
    
    print("Found {} face(s) in this photograph.".format(len(face_locations)))
    
    for count, face_coords in enumerate(face_locations):
        (top, right, bottom, left) = face_coords
        print("Face {} located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(count, top, left, bottom, right))

def outlineFacesInImage(img):
    face_locations = getCoordinates(img)
    
    for face_coords in face_locations:
        (top, right, bottom, left) = face_coords
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 225), 2)
    
    displayImage(img)

def displayIndividualFaces(list_of_faces):
    for face in list_of_faces:
        displayImage(face)


def displayImage (image):
    cv2.imshow('', image)
    cv2.waitKey()



student_img = cv2.imread('Photos_Directory' + '/' + 'students.jpg')

testFaceDetection(student_img)
#outlineFacesInImage(student_img)
facesPerImage = getListOfFaces(student_img)
#displayIndividualFaces(facesPerImage)


