import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# Color
color = (255, 0, 0)
# Create window
cv2.namedWindow("Frame")
# Find the drink!!
cls_id = 163
idx = 0
while True:
    ret, frame = cap.read()
    cv2.imwrite("./frame.png",frame)
    outputs = model.predict(source="./frame.png", conf=0.45,save=False) # treat predict as a Python generator
    print(next(outputs))
    print(type(outputs))
    cv2.imshow("Frame", frame)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()