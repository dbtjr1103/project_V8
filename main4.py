import cv2
from gui_buttons import Buttons
import math
from ultralytics import YOLO
import imutils
import itertools

model = YOLO("alc_v8.pt")
cap = cv2.VideoCapture('alc_clip.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280 )

outputs = model.predict('2023-01-17_Untitled.mp4', return_outputs=True, conf = 0.85) # treat predict as a Python generator

# Color
color = (255, 0, 0)

# Create window
cv2.namedWindow("Frame")


# Find the drink!!    
cls_id = 163
idx = 0

while True:
    ret, frame = cap.read()
    
    for output in outputs:
        idx += 1
        for i in range(len(output['det'])):
            if output['det'][i][5] == cls_id:
                # itertools.islice 는 generator의 찾고자 하는 index의 값을 불러와준다. 하지만 오류 발생
                (x,y,w,h) = int(next(itertools.islice(outputs, idx, None))['det'][0][0], next(itertools.islice(outputs, idx, None))['det'][0][1], next(itertools.islice(outputs, idx, None))['det'][0][2],  next(itertools.islice(outputs, idx, None))['det'][0][3])
        
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
