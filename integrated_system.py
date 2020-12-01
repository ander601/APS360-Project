#from image_cascadeClassifier import *
from mask_analysis import *
from video_integration import *
import cv2
import face_recognition
from detect_and_split import *
import numpy as np

def integrated_system():
    analyzer=mask_analysis_system()
    cap = cv2.VideoCapture(0)
    process_this_frame = True
    
    while True:
        _, frame = cap.read()
        
         # Resize frame of video to 1/4 size for faster face recognition processing
        scale_size = 4
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale_size, fy=1/scale_size)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            coordinate_list = getCoordinates(rgb_small_frame)
            face_list = getListOfFaces(rgb_small_frame)
            
            judge_list = analyzer.analysis(face_list)
            video_integration_with_scale(frame, coordinate_list, judge_list, scale_size)
        
        process_this_frame = not process_this_frame
       
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
            
        #this shows just your face
#        count=0
#        for face in face_list:
#            count=count+1
#            cv2.imshow('img'+str(count), face)


if __name__ == '__main__':
    integrated_system()

