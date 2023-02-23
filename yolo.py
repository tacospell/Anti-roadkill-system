import cv2
import numpy as np
i = 0

net = cv2.dnn.readNet("setting/yolov4.weights","setting/yolov4.cfg")
classes = []
with open("setting/coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

image = cv2.imread('image.jpg')
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

for out in outs:
 for detection in out:
  scores = detection[5:]
  class_id = np.argmax(scores)
  confidence = scores[class_id]
  if confidence > 0.5:
   center_x = int(detection[0] * width)
   center_y = int(detection[1] * height)
   w = int(detection[2] * width)
   h = int(detection[3] * height)
   x = int(center_x - w / 2)
   y = int(center_y - h / 2)
   
   font = cv2.FONT_HERSHEY_PLAIN
   color = colors[i]
   
   cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
   cv2.putText(image, str(classes[class_id]) , (x, y + 30), font, 3, color, 3)
   i = i+1
   break
   
cv2.imwrite('image2.jpg',image)
