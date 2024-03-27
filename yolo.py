import cv2 
from ultralytics import YOLO

# image path
img = "t.jpg"

# read image
img = cv2.imread(img)
 
# load model
face_model = YOLO("model/yolov8n-face.pt")

# predict
results = face_model.predict(img, conf=0.40)

# draw bounding box
for info in results:
    parameters = info.boxes
    for box in parameters:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = y2 - y1, x2 - x1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)