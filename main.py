from imutils.video import FPS
import numpy as np  
import argparse  
import imutils  
import time  
import cv2  
import os
 
ap = argparse.ArgumentParser()  
 
ap.add_argument("-i", "--input", type=str, help="input video")
ap.add_argument("-o", "--output", type=str, help="output video")  
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum confidence")
 
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold")
 
args = vars(ap.parse_args())
 
labelsPath = "YOLO-Object-Detection-master\yolo-coco\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
 
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
 
weightsPath = "YOLO-Object-Detection-master\yolo-coco\yolov3.weights"
configPath = "YOLO-Object-Detection-master\yolo-coco\yolov3.cfg"


print("starting...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
 
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
 
if not args.get("input", False):
    print("webcam start")
    vs = cv2.VideoCapture(0)
else:
    print("video start")
    vs = cv2.VideoCapture(args["input"])


fps = FPS().start()


writer = None
(W, H) = (None, None)
 
while True:
    ret, frame = vs.read()


    if args["input"] is not None and frame is None:
        break
   
    frame = imutils.resize(frame, width=1000)


    if W is None or H is None:
        (H, W) = frame.shape[:2]    
     
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
     
    net.setInput(blob)
    layerOutputs = net.forward(ln)
     
    boxes = []
    confidences = []
    classIDs = []    
     
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
             
            if confidence > args["confidence"]:
                 
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")  
                 
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                 
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
   
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
   
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
                         
            color = [int(c) for c in COLORS[classIDs[i]]]
                         
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{} : {:.2f}%".format(LABELS[classIDs[i]], confidences[i])
           
            y = y - 15 if y - 15 > 15 else y + 15        
             
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     
    cv2.imshow("Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
         
    if key == ord("s"):
        break
         
    fps.update()
         
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
         
    if writer is not None:
        writer.write(frame)
 
fps.stop()
print("[play time : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))
 
vs.release()
cv2.destroyAllWindows()




