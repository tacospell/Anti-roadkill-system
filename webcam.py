import cv2
import time

vc = cv2.VideoCapture(0)
start = time.time()

while True:
	cur_time = time.time() - start
	print("time is " + str(cur_time))
	
	ret, frame = vc.read()
	cv2.imshow("video", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('s'):
		print("capturing")
		cv2.imwrite('image.jpg',frame)
		break
