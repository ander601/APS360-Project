from image_cascadeClassifier import *
from mask_analysis import *
from video_integration import *
import cv2

def integrated_system():
    analyzer=mask_analysis_system()
    face_finder=cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        coordinate_list = getFaceCoordinates(img, face_finder)

        face_list = getListOfFaces(img, coordinate_list)
        judge_list = analyzer.analysis(face_list)
        video_integration(img, coordinate_list, judge_list)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        count=0
        for face in face_list:
            count=count+1
            cv2.imshow('img'+str(count), face)


if __name__ == '__main__':
    integrated_system()


