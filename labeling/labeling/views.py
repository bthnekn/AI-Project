from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
import base64
import cv2
import numpy as np
import mnist_test
def Anasayfa(request):
	return render(request, 'anasayfa.html', {})
def isle(request):
	if(request.POST):
		veri = request.POST.get('veri')
		veri = veri.replace("data:image/jpeg;base64,","")
		veri = veri.replace("data:image/jpg;base64,","")
		veri = veri.replace("data:image/png;base64,","")
		veri = veri.replace("data:image/gif;base64,","")
		veri = veri.replace("data:image/bmp;base64,","")
		im_bytes = base64.b64decode(veri)
		im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
		frame = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
		veri = process(frame)
		print(veri)
		return HttpResponse(str(veri))
def get_img_contour_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,3)
    thresh1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	    cv2.THRESH_BINARY,7,3.5)
    #thresh1 = cv2.erode(thresh1,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    thresh1 = (255-thresh1)
    return img, contours, thresh1
loaded_model = mnist_test.model()
def yazi(img):
	img, contours, thresh = get_img_contour_thresh(img)
	ans1 = ''
	if len(contours) > 0:
		contour = max(contours, key=cv2.contourArea)
	if cv2.contourArea(contour) > 2500:
	
		x, y, w, h = cv2.boundingRect(contour)
		newImage = thresh[y:y + h, x:x + w]
		newImage = cv2.resize(newImage, (28, 28))
		newImage = np.array(newImage)
		newImage = newImage.flatten()
		newImage = newImage.reshape(28,28,1)
		ans1 = loaded_model.predict(newImage)
		ans1=ans1.tolist()
		ans1 = ans1[0].index(max(ans1[0]))

	x, y, w, h = 0, 0, 300, 300
	return str(ans1)





CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3
COLORS = [(0, 255, 0)]

def process(frame):
	keys = []
	net = cv2.dnn.readNet("postit.weights", "yolov4-tiny.cfg")
	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(size=(416, 416), scale=1/255, swapRB=True,crop=False)
	classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for (classid, score, box) in zip(classes, scores, boxes):
		x,y,w,h = box[0],box[1],box[2],box[3]
		color = COLORS[int(classid) % len(COLORS)]
		roi = frame[y:y+h, x:x+w]
		keys.append(yazi(roi))
	return keys
