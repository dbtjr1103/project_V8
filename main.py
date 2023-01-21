import cv2
from gui_buttons import Buttons
import torch
import sys
import math


# Initialize Buttons
# 버튼 만들기
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("keyboard", 20, 180)
button.add_button("remote", 20, 260)
button.add_button("scissors", 20, 340)

colors = button.colors


# Opencv DNN
# DNN 모델 불러오기
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


# Load class lists
# 인식하고자 하는 클래스들을 불러와서 classes에 append 해준 리스트를 생성한다.
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# Initialize camera
# cv2.VideoCapture(0) 은 Webcam. 다른 영상을 재생하고 싶으면 괄호 안에 '영상 이름' 을 넣어주면 된다.
cap = cv2.VideoCapture(0)

# 캡쳐한 영상의 해상도를 조절해준다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 버튼 클릭을 인식해주는 함수 생성
def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

# Create window
# 영상을 재생할 창을 생성해주고, 클릭을 인식해주는 위 함수를 콜백해준다.
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)


# Detection
while cap.isOpened(): # 캡쳐한 영상을 실행
    
    # Get frames
    ret, frame = cap.read() # 캡쳐한 영상의 frame들을 불러와준다. 결국 영상도 이미지의 연속이기 때문.

    # Get active buttons list
    # 위에 생성한 버튼을 display 해준다
    active_buttons = button.active_buttons_list()
    print("Active buttons", active_buttons)

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.2, nmsThreshold=.4) # 인식한 객체의 class_id, score, bbox 좌표를 가져온다.
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox # bbox 좌표 지정
        class_name = classes[class_id] # 위에서 만든 classes list에서 class_id 의 index 로 class_name을 지정해준다
        color = colors[class_id] # 동일한 방법으로 색 지정

        if class_name in active_buttons: # 찾고자 하는 클래스가 생성한 버튼에 있다면
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            # 텍스트와 바운딩 박스를 그려준다
    
    # Show Text if located
    # w_divisor 와 h_divisor 는 위 해상도에 3을 나누어준 값. 이걸 통해서 화면을 몇 개의 그리드로 인식하게 해줄지 정해준다.
    w_divisor = 1280/3
    h_divisor = 720/3
    
    # 바운딩 박스에 센터를 tuple 변수로 지정해준다. 
    center_bbox = ((x+(w/2)), (y+(h/2)))
    
    # new 는 각 좌표를 divisor들로 나누어준 값을 올림하여 tuple로 생성해준 것. 이런 계산을 하게 되면 (1,1) 에서 (3,3) 까지 center_bbox 의 좌표를 알 수 있다.
    new = (math.ceil(center_bbox[0]/w_divisor), math.ceil(center_bbox[1]/h_divisor))
    if new == (1,1):
        cv2.putText(frame, 'Located in Upper Left Corner', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (2,1):
        cv2.putText(frame, 'Located in Upper Center', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    elif new == (3,1):
        cv2.putText(frame, 'Located in Upper Right Corner', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
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


    # Display buttons
    button.display_buttons(frame)
    
    # 프레임 재생
    cv2.imshow("Frame", frame)
    
    # q를 눌러 강제로 종료시켜줄 수 있다. 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

