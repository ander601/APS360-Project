import cv2

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
            label = "Mask"
        else:
            outline_color = (0, 0, 255) #blue
            label = "No Mask"
        
        cv2.rectangle(img, (left, top), (right, bottom), outline_color, 2)
        cv2.putText(img, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        index=index+1

    cv2.imshow('webcam', img)
    return

