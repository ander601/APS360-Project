import face_recognition
import cv2
import os
from mask_analysis import *


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
    if not face_locations:
        print("No faces found in this photograph.")
        return False
    
    print("Found {} face(s) in this photograph.".format(len(face_locations)))
    
    for count, face_coords in enumerate(face_locations):
        (top, right, bottom, left) = face_coords
        print("Face {} located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(count, top, left, bottom, right))

    return True

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
    
def show_results(img, face_list, mask_detection_list):
    font = cv2.FONT_HERSHEY_DUPLEX
    
    index=0
    for (top, right, bottom, left) in face_list:
        if mask_detection_list[index]==True:
            outline_color = (255, 0, 0) #red
            label = "Mask"
        else:
            outline_color = (0, 0, 255) #blue
            label = "No Mask"
        
        cv2.rectangle(img, (left, top), (right, bottom), outline_color, 2)
        cv2.putText(img, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        index=index+1

    cv2.imshow('', img)
    cv2.waitKey()
    return

if __name__ == '__main__':
    #img = cv2.imread('Photos_Directory' + '/' + 'students.jpg')
    
    img_path = 'Photos_Directory' + '/' + 'students.jpg'
    colored_img = cv2.imread(img_path)
    img = face_recognition.load_image_file(img_path)

    if testFaceDetection(img):
        #outlineFacesInImage(img)
        #facesPerImage = getListOfFaces(img)
        #displayIndividualFaces(facesPerImage)
        
        analyzer = mask_analysis_system()

        coordinate_list = getCoordinates(img)
        face_list = getListOfFaces(img)
        judge_list = analyzer.analysis(face_list)
        
        show_results(colored_img, coordinate_list, judge_list)


