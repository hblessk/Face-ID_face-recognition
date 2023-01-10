#락, 언락 판단


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'faces/'
# listdir(입력 경로 내의 모든 파일과 폴더명을 리스트 반환), isfile(파일이 있으면 true, 파일이 아니거나 없으면 false), join(경로, 파일명): 파일명과 경로 합치기
# in listdir(data_path): faces 폴더안의 파일들을 리스트로 만들어준 다음에/ 만약 join(data_path,f)가 파일 이라면(isfile) f로 받아서 onlyfiles 리스트에 넣어줌
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
# 각 빈리스트로 만들어줌
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    # IMREAD_GRAYSCALE: 비록 해당 원본 이미지가 RGB의 칼라 이미지이지만, Gray 색상으로 해석해, 이미지 객체를 반환
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # 내 얼굴 이미지를 받아들이는 함수
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
# Training_Data(data type)이 설정 되어 있다면, 데이터 형태가 다를 경우에만 복사(copy)됨
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")
# 얼굴 검출하려고 미리 학습시켜놓은 XML 포맷으로 저장된 분류기 불러옴
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):     # 얼굴 인식 함수 선언
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지에서 얼굴을 검출함
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]
    # 얼굴이 검출되면 얼굴 위치에 대한 좌표 정보 리턴 받음
    for(x,y,w,h) in faces:
        # 원본 이미지에 얼굴의 위치를 표시함
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        # 검출된 얼굴은 영역 내부에서만 진행하려고 ROI 실행
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)

while True:
    # 카메라로 부터 사진 한장 읽기
    ret, frame = cap.read()
    # 얼굴 검출 시도
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   #검출된 사진을 흑백으로 변환
        result = model.predict(face)    # 학습한 모델로 예측하기
        # result[1] :신뢰도이고 0에 가까울수록 자신과 같다는 뜻
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            # 유사도 화면에 표시
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        # 78 보다 크면 동일 인물로 간주해 UnLocked!
        if confidence > 82:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:         # 얼굴을 찾을 수 없을 때 "Face Not Found"
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:   # waitKey의 리턴 값은 키보드로 입력한 키와 동일한 아스키코드의 값(enter키:13)
        break


cap.release()
cv2.destroyAllWindows()

