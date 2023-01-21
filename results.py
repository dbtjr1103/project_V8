from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

# 간단 결과 출력. opencv를 사용하기 때문에 빠르게 모델 학습도만 보고자 할 때 사용
model = YOLO('dairy10_1.engine')

results = model.predict(source='0', show = True)

print(*results)