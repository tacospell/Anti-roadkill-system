import cv2
import time

vc = cv2.VideoCapture(0)
start = time.time()

while True:
	cur_time = time.time() - start
	print("time is " + str(cur_time))
	
	ret, frame = vc.read()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.putText(frame, "I need Help", (50, 100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
	cv2.imshow("video", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('s'):
		print("capturing")
		cv2.imwrite('image.jpg',frame)
		break
