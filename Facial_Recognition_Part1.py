#카메라 로 얼굴 사진저장(100장)

import cv2


# 객체 생성
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 정면 얼굴 인식


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)    # 입력영상에서 얼굴을 검출

    if faces is():
        return None

    # 각각의 행마다 (x,y,w,h) 받아와서 사각형을 그리는 코드
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read() # 재생되는 비디오의 한 프레임씩 읽음/ 제대로 프레임을 읽으면 ret값이 True, 실패하면 False/ frame에 읽은 프레임이 나옵니다
    # 얼굴 감지 하여 얼굴만 가져오기
    if face_extractor(frame) is not None:
        count+=1
        # 얼굴 이미지 크기를 200x200으로 조정
        face = cv2.resize(face_extractor(frame),(200,200))
        # 조정된 이미지를 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)      # cv2.cvtColor: 프레임을 흑백으로 변환
        # faces폴더에 jpg파일로 저장
        file_name_path = 'faces/user'+str(count)+'.jpg'
        # 변환된 이미지나 동영상의 특정 프레임을 저장(저장될 파일명, 저장할 이미지)
        cv2.imwrite(file_name_path,face)
        # 화면에 얼굴과 현재 저장 개수 표시(50,50)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)     # 폰트 종류
        cv2.imshow('Face Cropper',face) # 이미지를 사이즈에 맞게 보여줌
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()   # 오픈한 cap 객체를 해제하는 것
cv2.destroyAllWindows()     # 완성한 모든 윈도우를 제거
print('Colleting Samples Complete!!!')