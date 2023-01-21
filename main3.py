from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import tensorflow as tf
import cv2

# Yolo 대신 tensorflow 를 사용하여 모델 불러오기
model = tf.keras.models.load_model('dairy10_1.pt')
results = model.predict('KakaoTalk_20230113_165709155.mp4', show = True)

# bbox 불러오기 -> 불가. 오류 
boxes = results[:, :, :4]
print(boxes)
print(*results)

# is.Opened() 대신 True를 사용하여 error 가 뜨지 안흔ㄴ 동안은 영상을 계속 재생해주는 방법
while True():
    # Get frames
    ret, frame = cap.read()
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

    