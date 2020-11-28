import cv2


def test_video_integration():
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        video_integration(img, faces, [False, False, False, False])
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()

def video_integration(img, face_list, mask_detection_list):
    index=0
    for (x, y, w, h) in face_list:
        if mask_detection_list[index]==True:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        index=index+1

    cv2.imshow('img', img)
    return
    
def video_integration_with_scale(img, face_list, mask_detection_list, scale_size):
    font = cv2.FONT_HERSHEY_DUPLEX
    
    index=0
    for (top, right, bottom, left) in face_list:
        top *= scale_size
        right *= scale_size
        bottom *= scale_size
        left *= scale_size

        if mask_detection_list[index]==True:
            outline_color = (255, 0, 0) #red
            label = "No mask"
        else:
            outline_color = (0, 0, 255) #blue
            label = "mask"
        
        cv2.rectangle(img, (left, top), (right, bottom), outline_color, 2)
        cv2.putText(img, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        index=index+1

    cv2.imshow('webcam', img)
    return


'''
        if mask_detection_list[index]==True:
            cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (top, left), (right, bottom), (0, 0, 255), 2)
        index=index+1
'''
