import cv2
import numpy as np
from os import listdir
from os import makedirs
from os.path import isdir, isfile, join
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




# 얼굴 검출 함수
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face

# 얼굴만 저장하는 함수
def take_pictures(name):
    
    

    # 해당 이름의 폴더가 없다면 생성
    if not isdir(name):
        makedirs(name)

    # 카메라 ON
    
    cap = cv2.VideoCapture(0)
    count = 0
    face_valid_notfound = False
    face_valid_found = False
    while True:
        
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면
        
        if face_extractor(frame) is not None:
            face_valid_found = True
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame),(200,200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = name + '/user'+str(count)+'.jpg' #요기
            cv2.imwrite(file_name_path,face) #요기

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        
        # 얼굴 사진 100장을 다 얻었거나 enter키 누르면 종료
        if cv2.waitKey(1)==13 or count==50:
            break
        
        

    cap.release()
    cv2.destroyAllWindows()
    
    print('Colleting Samples Complete!!!')

#새로운 유저 사진찍기
if __name__ == "__main__":
    # 사진 저장할 이름을 넣어서 함수 호출
    count = 0
    file_name = 'user'
    file_path ='faces/%s%d' %(file_name,count)         
    while os.path.exists(file_path):
        file_path ='faces/%s%d' %(file_name,count)  
        count+=1
    
    take_pictures(file_path)


# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    # 학습된 모델을 파일로 저장
    model.save('trainedmodels/' + name + '.xml')
    
    #학습 모델 리턴
    return model

# 여러 사용자 학습

#faces 폴더의 하위 폴더를 학습
data_path = 'faces/'
# 폴더만 색출
model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
#학습 모델 저장할 딕셔너리
models = {}
   # 각 폴더에 있는 얼굴들 학습
for model in model_dirs:
    print('model :' + model)
    # 학습 시작
    result = train(model)
    # 학습이 안되었다면 패스!
    if result is None:
        continue
    # 학습되었으면 저장
    print('model2 :' + model)
    models[model] = result

