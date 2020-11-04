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

test_video_integration()

'''
        if mask_detection_list[index]==True:
            cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (top, left), (right, bottom), (0, 0, 255), 2)
        index=index+1
'''