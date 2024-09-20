import numpy as np
import cv2
from recog import face_detector

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용

while(1):
    ret, frame = cap.read() # 사진 촬영
    image, face = face_detector(frame)
    try:
        cv2.imshow('camera',image)
    except:
        pass
    
cap.release()
cv2.destroyAllWindows()