import face_recognition
import cv2
import os
from mask_analysis import *

#returns a list of coordinates; 1 set for each face in the photo.
def getCoordinates(img):
    return face_recognition.face_locations(img)

def getListOfFaces(img):
    face_locations = getCoordinates(img)
    
    if not face_locations:
        print("No faces found in this photograph.")
        return []
        
    print("Found {} face(s) in this photograph.".format(len(face_locations)))

    width = height = 244
    list_of_faces = []
    
    for count, face_coords in enumerate(face_locations):
        (top, right, bottom, left) = face_coords
        
        face_image = img[top:bottom, left:right]
        resized = cv2.resize(face_image, (width, height))
        
        list_of_faces.append(resized)
        
        print("Face {} located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(count, top, left, bottom, right))

    return list_of_faces

    
def outline_faces(img, face_list, mask_detection_list):
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

    return img


if __name__ == '__main__':

    directory = 'Photos_Directory/'

    for filename in os.listdir(directory):
        
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg")):
            continue
            
        img_path = directory + filename
        
        colored_img = cv2.imread(img_path)
        grey_scaled_img = face_recognition.load_image_file(img_path)

        face_list = getListOfFaces(grey_scaled_img)
        if face_list:
            
            analyzer = mask_analysis_system()

            coordinate_list = getCoordinates(grey_scaled_img)
            judge_list = analyzer.analysis(face_list)
            
            marked_img = outline_faces(colored_img, coordinate_list, judge_list)

            cv2.imshow(filename, marked_img)
            
    cv2.waitKey()








