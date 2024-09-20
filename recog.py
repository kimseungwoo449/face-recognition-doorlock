import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import RPi.GPIO as GPIO
import time
import os

#relay 21 ultrasonic 24 buzzer 18



    
# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

def buzzer():
    buzzer = 18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer,GPIO.OUT)
    pwm=GPIO.PWM(buzzer,262)
    #부저 2초간 울리기
    pwm.start(50.0)
    time.sleep(2)
    pwm.stop()

def relay():
    relay = 21
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(relay,GPIO.OUT)
    GPIO.output(relay, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(relay, GPIO.LOW)
    time.sleep(5)
    #GPIO.output(relay, GPIO.HIGH)
    #time.sleep(0.1)
    #GPIO.output(relay, GPIO.LOW)
    
    
def sonic() :
    GPIO_TRIGGER = 24
    GPIO_ECHO    = 24
    GPIO.setmode(GPIO.BCM)
    print ("거리측정")
    while(1):
        
        # Set pins as output and input

        GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger

        # Set trigger to False (Low)

        GPIO.output(GPIO_TRIGGER, False)

        # Allow module to settle

        time.sleep(0.5)

        # Send 10us pulse to trigger

        GPIO.output(GPIO_TRIGGER, True)

        time.sleep(0.00001)

        GPIO.output(GPIO_TRIGGER, False)

        start = time.time()

        GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo 

        while GPIO.input(GPIO_ECHO)==0:

          start = time.time()

        while GPIO.input(GPIO_ECHO)==1:

          stop = time.time()

        # Calculate pulse length
        elapsed = stop-start
        # Distance pulse travelled in that time is time

        # multiplied by the speed of sound (cm/s)

        distance = elapsed * 34300
        # That was the distance there and back so halve the value

        distance = distance / 2

        time.sleep(0.5)

        if (distance <= 100):
            break
            

    print ("Distance : %.1fcm" % distance)
    
    return distance

#학습모델된 모델들 불러오기
def read():
    #xml파일 위치
    models_path = 'trainedmodels/'
    #models_path의 위치에있는 xml파일의 이름들을 리스트로 만듦
    trained_models_list = [f for f in listdir(models_path) if f.endswith('.xml')]
    #학습모델이 들어갈 딕셔너리 선언
    models = {}
    #각 학습모델들을 딕셔너리에 넣기
    for model in trained_models_list:
        trained_models = cv2.face.LBPHFaceRecognizer_create()
        trained_models.read('trainedmodels/'+ model)
        result = trained_models
        models[model]=result

    return models

#얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),1)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

# 인식 시작
def run(models):    
    #카메라 열기 
    cap = cv2.VideoCapture(0)
    face_valid_unlocked = False
    face_valid_locked = False
    face_valid_missing = False
    while True:
        #카메라로 부터 사진 한장 읽기 
        ret, frame = cap.read()
        
        # 얼굴 검출 시도 
        image, face = face_detector(frame)
        nomal_time = time.time()
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            #검출된 사진을 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #위에서 학습한 모델로 예측시도
            for key, value in models.items():
                result = value.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    
            #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.         
            if min_score < 500:
                #????? 어쨋든 0~100표시하려고 한듯 
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            
            if confidence > 79:
                if not face_valid_unlocked:
                    start_time_unlocked = time.time()
                    face_valid_unlocked = True
                    if face_valid_locked:
                        face_valid_locked = False
                    if face_valid_missing:
                        face_valid_missing = False  
                
                nomal_time = time.time()
                # 1초간 80이상 시 언락
                if nomal_time - start_time_unlocked <= 1:
                    cv2.putText(image, "locked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,255), 2)
                    cv2.imshow('Face Cropper', image)
                else:
                    cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', image)
                    print('인식완료. 잠금해제')
                    #relay 5sec
                    relay()
                    break

            # 79 이하일 때, 인식 X
            else:
                if not face_valid_locked:
                    start_time_locked = time.time()
                    face_valid_locked = True
                    if face_valid_unlocked:
                        face_valid_unlocked = False
                    if face_valid_missing:
                        face_valid_missing = False     

                nomal_time = time.time()
                # 3초간 80미만 시 저장
                if nomal_time - start_time_locked <= 3:
                    cv2.putText(image, "locked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,255), 2)
                    cv2.imshow('Face Cropper', image)
                else:
                    cv2.putText(image, "locked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,255), 2)
                    cv2.imshow('Face Cropper', image)
                    count = 0
                    file_name = 'visitor'
                    file_ext = '.jpg'
                    visitor_path ='static/visitors/%s%d%s' %(file_name,count,file_ext)
                    
                    while os.path.exists(visitor_path):
                        visitor_path = 'static/visitors/%s%d%s'%(file_name,count,file_ext)    
                        count+=1
                    
                    
                    cv2.imwrite(visitor_path,face)
                    #방문자 발생시 부저울림
                    buzzer()

                    break
             
        except:
            #얼굴 검출 안됨 
            if not face_valid_missing:
                start_time_missing = time.time()
                face_valid_missing = True
                if face_valid_unlocked:
                    face_valid_unlocked = False   
                if face_valid_locked:
                    face_valid_locked = False
                        
            
            # 10초간 얼굴 없을 시 종료
            if nomal_time - start_time_missing <= 10:
                #얼굴 검출 안됨 
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Cropper', image)
            else:
                cv2.putText(image, "locked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,255), 2)
                cv2.imshow('Face Cropper', image)
                print('10초간 얼굴이 감지되지 않았습니다.')
                break 
            
            pass
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 학습 시작
    # 고!
    while(1):
        
        models = read()
        distance = 1000000
        distance = sonic() 
        if distance <= 100:
            run(models)
            time.sleep(3) #3초후 실행    
    GPIO.cleanup() #라즈베리파이로 확인

    
