from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-obb.pt')
model = YOLO('best.pt')

# Define path to the image file
source = './test1.jpg'
#source = cap

# Run inference on the source
#results = model(source, stream=True)  # list of Results objects

results = model(source, save=True)

for r in results:
	x = r.obb.xywhr[0][0].numpy()
	y = r.obb.xywhr[0][1].numpy()
	w = r.obb.xywhr[0][2].numpy()
	h = r.obb.xywhr[0][3].numpy()
	r = r.obb.xywhr[0][4].numpy()
	print("x: ",x,"y: ",y,"w: ",w,"h: ",h,"r: ",r)



