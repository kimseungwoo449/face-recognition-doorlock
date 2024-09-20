import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
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

