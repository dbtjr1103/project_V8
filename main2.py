import cv2
from gui_buttons import Buttons
import math
from ultralytics import YOLO
import imutils

# 모델 불러오기
model = YOLO("alc_v8.pt")
cap = cv2.VideoCapture('alc_clip.mp4')
#해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280 )
outputs = model.predict('alc_clip.mp4', return_outputs=True, conf = 0.90) # treat predict as a Python generator

# Color
color = (255, 0, 0)

# Create window
cv2.namedWindow("Frame")


while True:
    # Get frames
    ret, frame = cap.read()
    # outputs = model.predict('alc_clip.mp4', return_outputs=True) # treat predict as a Python generator
    
    # 찾고자 하는 객체의 class 값 입력
    find_class = 95
    
    # For 문 사용 대신 generator 에 들어간 array들을 뽑아 올 수 있음
    # array는 이중 array 로 [bbox_x, bbox_y, bbox_w, bbox_h, score, cls_id] 가 담겨져 있음
    detect_Result=next(outputs)['det']
    print(detect_Result)
    
    # 객체 인식을 성공 하고 and 찾고자 하는 cls_id 가 존재할 경우에
    if len(detect_Result) > 0 and (True in (list(map(lambda x: find_class == x[5], detect_Result)))):        
        
        # 그 행의 값을 idx 로 설정해준다    
        idx = (list(map(lambda x: find_class == x[5], detect_Result))).index(True)
        # bbox 좌표들 불러와서 tuple 형태로 저장
        (x,y,w,h) = int((detect_Result)[idx][0]),int((detect_Result)[idx][1]),int((detect_Result)[idx][2]),int((detect_Result)[idx][3])
        # bbox 그려주기
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # main.py 참고
    w_divisor = 1280/3
    h_divisor = 720/3
    center_bbox = ((x+(w/2)), (y+(h/2)))
    new = (math.ceil(center_bbox[0]/w_divisor), math.ceil(center_bbox[1]/h_divisor))
    if new == (1,1):
        cv2.putText(frame, 'Located in Upper Left Corner', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (2,1):
        cv2.putText(frame, 'Located in Upper Center', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (3,1):
        cv2.putText(frame, 'Located in Upper Right Center', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (1,2):
        cv2.putText(frame, 'Located in Left', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (2,2):
        cv2.putText(frame, 'Located in Center', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (3,2):
        cv2.putText(frame, 'Located in Right', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (1,3):
        cv2.putText(frame, 'Located in Bottom Left Corner', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (2,3):
        cv2.putText(frame, 'Located in Bottom Center', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (3,3):
        cv2.putText(frame, 'Located in Bottom Right Corner', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

