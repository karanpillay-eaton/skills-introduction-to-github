# import the necessary packages
import math
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
#from image_process.motion_detection.SingleMotionDetector import SingleMotionDetector
from flask import Flask,send_from_directory

import cvzone
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import os
import socket
import logging

logger = logging.getLogger(__name__)
log_file_path = os.getcwd()+'/logs/Analytics_Usage_V5.log'
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



host = 'local host'
sock_comm_port = 13000

DOWNLOAD_DIRECTORY = os.getcwd()+'/static'
DOWNLOAD_DIRECTORY_SETUP = os.getcwd()+'/setup'

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# initialize the video stream and allow the camera sensor to
model = YOLO("../Yolo-Weights/yolov5nu.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# warmup
NMS_THRESHOLD=0.5
MIN_CONFIDENCE=0.8

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [ layer_name [i - 1] for i in model.getUnconnectedOutLayers()]
src ='rtsp://192.168.4.15:8554/mjpeg/1'

vs = VideoStream(src).start()
writer = None
time.sleep(0.5)
totalCountUp = []
line_number2 = 2
f = open("setup/readme.txt", "r")
file_buf = f.readlines()
f.close()
line2 = file_buf[line_number2 - 1]
file_buf = line2  # @SETX1 80 Y1 180 X2 200 Y2 90 X3 360 Y3 70 X4 400 Y4 190 X5 300 Y5 200 #
print(line2)
valid_values =[]
for value in file_buf.split():
    if value.isdigit():
        valid_values.append(int(value))

array_size =len(valid_values)
print(len(valid_values))
print(array_size)


# for  shape
line_number = 1
f1 = open("setup/readme.txt", "r")
lines = f1.readlines()
f1.close()
line = lines[line_number - 1]  # line=pentagon
shape = ""
for l in line:
    shape += l
    value = shape
value = (shape)
print(value)

line_number1 = 3
f2 = open("setup/readme.txt", "r")
lines = f2.readlines()
f2.close()
line1 = lines[line_number1 - 1]

user = int(line1)  # user=2
print(user)
print(type(user))

coordinates = np.empty(array_size, dtype=int)
i =0
for z in file_buf.split():
    if z.isdigit():
        coordinates[i] = z

        i += 1
print(coordinates)
print(len(coordinates))
limitsUp = [coordinates[0], coordinates[1]]
limitsDown = [coordinates[2], coordinates[3]]



@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route('/live')
def live():
    return render_template("live.html")
def pedestrian_detection(frame, model, layer_name, personidz=0):
	(H, W) = frame.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				#print("Center x and Y")
				#print(centerX)
				#print(centerY)
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates

			#print("ID: ", i)
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h, i), centroids[i])
			results.append(res)
	# return the list of results
	return results
def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock
	global reload_request
	global intrusion_id
	global mask
	cam_err = False
	cam_err_ctr = 0
	reload_request = 0
	line_cross_flag = 0
	f2 = open("data/Intrusion_ID.txt", "r")
	file_buf2 = f2.read()
	f2.close()
	
	intrusion_id = int(file_buf2)
	logger.info('\nAnalytics Code in detect motion\n')
	
	#print(intrusion_id)
	
	tempstr = "App Status Usage Mode"
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.connect(('127.0.0.1', sock_comm_port))
	s.send(tempstr.encode())
	s.close()
	
    # loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		timestamp = datetime.datetime.now()
		if frame is None :
			if(cam_err == False):            
				logger.info('\nCamera Not ready\n')
				cam_err = True
				msg = "App Status Camera not ready"
				s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
				s.connect(('127.0.0.1', sock_comm_port))
				s.send(msg.encode())
				s.close()
			cam_err_ctr = cam_err_ctr + 1
			if(cam_err_ctr > 500) :
				vs.stop()
				time.sleep(0.1)
				vs = VideoStream(src).start()
				logger.info('\nCamera Reconnecting\n')
				cam_err_ctr = 0
			continue
        
		frame = imutils.resize(frame, width=800)
		if(cam_err == True):
			cam_err = False
			cam_err_ctr = 0
			tempstr = "App Status Usage Mode"
			s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
			s.connect(('127.0.0.1', sock_comm_port))
			s.send(tempstr.encode())
			s.close()
        # if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		#imgRegion = cv2.bitwise_and(frame, mask)
		cv2.putText(frame,timestamp.strftime("%d/%m/%Y %H:%M:%S"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
		imgRegion = cv2.bitwise_and(img, mask)

		imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
		img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
		results = model(imgRegion, stream=True)


		detections = np.empty((0, 5))
		
						
		if value == "circle\n":
			cv2.circle(frame, (int((limitsUp[0] + limitsDown[0]) / 2), int((limitsUp[1] + limitsDown[1]) / 2)),
                   int(abs(limitsDown[0] - limitsUp[0]) / 2), (0, 0, 255), 4)

			for r in results:
				boxes = r.boxes
				for box in boxes:
					# Bounding Box
					x1, y1, x2, y2 = box.xyxy[0]
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					# cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
					w, h = x2 - x1, y2 - y1

					# Confidence
					conf = math.ceil((box.conf[0] * 100)) / 100
					# Class Name
					cls = int(box.cls[0])
					currentClass = classNames[cls]

					if currentClass == "person" and conf > 0.3:
						# cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
						#                    scale=0.6, thickness=1, offset=3)
						# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
						currentArray = np.array([x1, y1, x2, y2, conf])
						detections = np.vstack((detections, currentArray))

			resultsTracker = tracker.update(detections)

			cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
			cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

			for result in resultsTracker:
				x1, y1, x2, y2, id = result
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				# print(result)
				w, h = x2 - x1, y2 - y1
				# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
				# cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
				# scale=2, thickness=3, offset=10)

				cx, cy = x1 + w // 2, y1 + h
				cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
				cv2.circle(frame, (int((limitsUp[0] + limitsDown[0]) / 2), int((limitsUp[1] + limitsDown[1]) / 2)),
						   int(abs(limitsDown[0] - limitsUp[0]) / 2), (0, 0, 255), 4)

				if limitsUp[0] <cx< limitsDown[0] and limitsUp[1] - 60 < cy < limitsDown[1] + 60:
					#if totalCountUp.count(id) == 0:
						cv2.circle(frame, (int((limitsUp[0] + limitsDown[0]) / 2), int((limitsUp[1] + limitsDown[1]) / 2)),int(abs(limitsDown[0] - limitsUp[0]) / 2), (0, 255, 0), 4)
						if (line_cross_flag == 0):
							line_cross_flag = 1
							intrusion_id += 1
							image_path = "static/" + str(intrusion_id) + ".png"
							cv2.imwrite(image_path, frame)
							tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
							s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
							s.connect(('127.0.0.1', sock_comm_port))
							s.send(tempstr.encode())
							s.close()
							time.sleep(2.0)
						else:

							line_cross_flag = 0
		elif value == "polygon\n":
			if len(coordinates) == 12:
				pts = np.array(
					[[coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]],
					 [coordinates[4], coordinates[5]],
					 [coordinates[6], coordinates[7]], [coordinates[8], coordinates[9]],
					 [coordinates[10], coordinates[11]]], np.int32)
				cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

				for r in results:
					boxes = r.boxes
					for box in boxes:
						# Bounding Box
						x1, y1, x2, y2 = box.xyxy[0]
						x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
						# cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
						w, h = x2 - x1, y2 - y1

						# Confidence
						conf = math.ceil((box.conf[0] * 100)) / 100
						# Class Name
						cls = int(box.cls[0])
						currentClass = classNames[cls]

						if currentClass == "person" and conf > 0.3:
							# cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
							#                    scale=0.6, thickness=1, offset=3)
							# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
							currentArray = np.array([x1, y1, x2, y2, conf])
							detections = np.vstack((detections, currentArray))

				resultsTracker = tracker.update(detections)



				for result in resultsTracker:
					x1, y1, x2, y2, id = result
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					# print(result)
					w, h = x2 - x1, y2 - y1
					# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
					# cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
					# scale=2, thickness=3, offset=10)

					cx, cy = x1 + w // 2, y1 + h
					cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

					if id <= user:
						frame = cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

						for r in results:
							boxes = r.boxes
							for box in boxes:
								# Bounding Box
								x1, y1, x2, y2 = box.xyxy[0]
								x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
								# cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
								w, h = x2 - x1, y2 - y1

								# Confidence
								conf = math.ceil((box.conf[0] * 100)) / 100
								# Class Name
								cls = int(box.cls[0])
								currentClass = classNames[cls]

								if currentClass == "person" and conf > 0.3:
									# cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
									#                    scale=0.6, thickness=1, offset=3)
									# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
									currentArray = np.array([x1, y1, x2, y2, conf])
									detections = np.vstack((detections, currentArray))

						resultsTracker = tracker.update(detections)



						for result in resultsTracker:
							x1, y1, x2, y2, id = result
							x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
							# print(result)
							w, h = x2 - x1, y2 - y1
							# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
							# cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
							# scale=2, thickness=3, offset=10)

							cx, cy = x1 + w // 2, y1 + h
							cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
					# print(id)

					else:
						if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
							cv2.polylines(frame, [pts], True, (0, 255,0), 4)
							if (line_cross_flag == 0):
								line_cross_flag = 1
								intrusion_id += 1
								image_path = "static/" + str(intrusion_id) + ".png"
								cv2.imwrite(image_path, frame)
								tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
								s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
								s.connect(('127.0.0.1', sock_comm_port))
								s.send(tempstr.encode())
								s.close()
								time.sleep(2.0)
						else:

							line_cross_flag = 0
			if len(coordinates) == 10:
				pts = np.array(
					[[coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]],
					 [coordinates[4], coordinates[5]],
					 [coordinates[6], coordinates[7]], [coordinates[8], coordinates[9]]], np.int32)
				cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

				for r in results:
					boxes = r.boxes
					for box in boxes:
						# Bounding Box
						x1, y1, x2, y2 = box.xyxy[0]
						x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
						# cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
						w, h = x2 - x1, y2 - y1

						# Confidence
						conf = math.ceil((box.conf[0] * 100)) / 100
						# Class Name
						cls = int(box.cls[0])
						currentClass = classNames[cls]

						if currentClass == "person" and conf > 0.3:
							# cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
							#                    scale=0.6, thickness=1, offset=3)
							# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
							currentArray = np.array([x1, y1, x2, y2, conf])
							detections = np.vstack((detections, currentArray))

				resultsTracker = tracker.update(detections)
				for result in resultsTracker:
					x1, y1, x2, y2, id = result
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					# print(result)
					w, h = x2 - x1, y2 - y1
					# cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
					# cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
					# scale=2, thickness=3, offset=10)

					cx, cy = x1 + w // 2, y1 + h
					cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

					if id <= user:
						cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

						for res in results:
							cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

							id = res[1][4]
							x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
							a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
							x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
							a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

							w, h = x2 - x1, y2 - y1  # width and height of the bounding box
							l, m = b1 + y1, b2 + y2
							cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
							ca, cb = a1 - w // -2, b1 - h
							cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
							cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

							cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
							cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)
					# print(id)

					else:
						if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
							cv2.polylines(frame, [pts], True, (0, 255,0), 4)
							if (line_cross_flag == 0):
								line_cross_flag = 1
								intrusion_id += 1
								image_path = "static/" + str(intrusion_id) + ".png"
								cv2.imwrite(image_path, frame)
								tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
								s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
								s.connect(('127.0.0.1', sock_comm_port))
								s.send(tempstr.encode())
								s.close()
								time.sleep(2.0)
						else:

							line_cross_flag = 0

			if len(coordinates) == 8:
				pts = pts = np.array([[coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]],
									  [coordinates[4], coordinates[5]],
									  [coordinates[6], coordinates[7]]], np.int32)
				frame = cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

				for res in results:  # results list consist of the person prediction probability,
					# bounding box coordinates, and the centroid
					cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

					id = res[1][4]
					x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
					a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

					w, h = x2 - x1, y2 - y1  # width and height of the bounding box
					l, m = b1 + y1, b2 + y2
					cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
					ca, cb = a1 - w // -2, b1 - h
					cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
					cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

					cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
					cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)

					if id <= user:
						cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

						for res in results:
							cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

							id = res[1][4]
							x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
							a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
							x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
							a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

							w, h = x2 - x1, y2 - y1  # width and height of the bounding box
							l, m = b1 + y1, b2 + y2
							cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
							ca, cb = a1 - w // -2, b1 - h
							cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
							cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

							cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
							cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)
					# print(id)

					else:
						if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
							cv2.polylines(frame, [pts], True, (0, 255,0), 4)
							if (line_cross_flag == 0):
								line_cross_flag = 1
								intrusion_id += 1
								image_path = "static/" + str(intrusion_id) + ".png"
								cv2.imwrite(image_path, frame)
								tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
								s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
								s.connect(('127.0.0.1', sock_comm_port))
								s.send(tempstr.encode())
								s.close()
								time.sleep(2.0)
						else:

							line_cross_flag = 0
			if len(coordinates) == 6:
				pts = pts = np.array(
					[[coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]],
					 [coordinates[4], coordinates[5]]], np.int32)
				image = cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

				for res in results:  # results list consist of the person prediction probability,
					# bounding box coordinates, and the centroid
					cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

					id = res[1][4]
					x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
					a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

					w, h = x2 - x1, y2 - y1  # width and height of the bounding box
					l, m = b1 + y1, b2 + y2
					cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
					ca, cb = a1 - w // -2, b1 - h
					cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
					cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

					cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
					cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)

					if id <= user:
						cv2.polylines(image, [pts], True, (0, 0, 255), 4)

						for res in results:
							cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

							id = res[1][4]
							x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
							a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
							x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
							a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

							w, h = x2 - x1, y2 - y1  # width and height of the bounding box
							l, m = b1 + y1, b2 + y2
							cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
							ca, cb = a1 - w // -2, b1 - h
							cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
							cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

							cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
							cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)
					# print(id)

					else:
						if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
							cv2.polylines(frame, [pts], True, (0, 255,0), 4)
							if (line_cross_flag == 0):
								line_cross_flag = 1
								intrusion_id += 1
								image_path = "static/" + str(intrusion_id) + ".png"
								cv2.imwrite(image_path, frame)
								tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
								s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
								s.connect(('127.0.0.1', sock_comm_port))
								s.send(tempstr.encode())
								s.close()
								time.sleep(2.0)
						else:

							line_cross_flag = 0
			if len(coordinates) == 4:
				pts = np.array([[coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]]],np.int32)
				image = cv2.polylines(frame, [pts], True, (0, 0, 255), 4)

				for res in results:  # results list consist of the person prediction probability,
					# bounding box coordinates, and the centroid
					cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

					id = res[1][4]
					x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
					a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

					w, h = x2 - x1, y2 - y1  # width and height of the bounding box
					l, m = b1 + y1, b2 + y2
					cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
					ca, cb = a1 - w // -2, b1 - h
					cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
					cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

					cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
					cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)

					if id <= user:
						cv2.polylines(image, [pts], True, (0, 0, 255), 4)

						for res in results:
							cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

							id = res[1][4]
							x1, y1, x2, y2 = res[1][0], res[1][1], res[1][2], res[1][3]
							a1, b1, a2, b2 = res[1][0], res[1][1], res[1][2], res[1][3]
							x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
							a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)

							w, h = x2 - x1, y2 - y1  # width and height of the bounding box
							l, m = b1 + y1, b2 + y2
							cx, cy = x1 + w // 2, y1 + h  # calculates the centroid coordinates (cx, cy) of the bounding box based on the top-left corner coordinates and the dimensions
							ca, cb = a1 - w // -2, b1 - h
							cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
							cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
								   scale=2, thickness=3, offset=10)

							cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
							cv2.circle(frame, (ca, cb), 5, (255, 0, 255), cv2.FILLED)
					# print(id)

					else:
						if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
							cv2.polylines(frame, [pts], True, (0, 255,0), 4)
							if (line_cross_flag == 0):
								line_cross_flag = 1
								intrusion_id += 1
								image_path = "static/" + str(intrusion_id) + ".png"
								cv2.imwrite(image_path, frame)
								tempstr = "Intrusion ID " + str(intrusion_id) + timestamp.strftime(
								" at %d/%m/%Y %H:%M:%S\r")
								s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
								s.connect(('127.0.0.1', sock_comm_port))
								s.send(tempstr.encode())
								s.close()
								time.sleep(2.0)
						else:

							line_cross_flag = 0
				
		with lock:
			outputFrame = frame.copy()
		

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				print("Image not ready\n")
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')
			

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")





@app.route('/intrusion/<path:path>',methods = ['GET','POST'])
def intrusion(path):
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, path, as_attachment=False)
    except FileNotFoundError:
        abort(404)



# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	logger.info('\nStarting Analytics Usage App\n')
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=False,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=False,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=1,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	
	t.daemon = True
	t.start()
	
	# start the flask app
	app.run(host='0.0.0.0', port='5000', debug=True,
		threaded=True, use_reloader=False)
	
# release the video stream pointer
vs.stop()
