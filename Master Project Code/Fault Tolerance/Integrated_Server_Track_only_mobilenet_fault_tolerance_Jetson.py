from __future__ import division
import sys
import os
import time
import numpy as np
import cv2
import cozmo
import socket
import tensorflow as tf

from PIL import Image
from cozmo.util import degrees
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

global frame 
frame = "current.jpg"
PORT = 2000

def preprocess_image(pil_image):
	img = pil_image.resize((224, 224))
	img = img_to_array(img)
	img = img.reshape(1, 224, 224, 3)
	img = img.astype('float32')
	img = preprocess_input(img)
	return img

def region_to_bbox(region, center=True):
	x = region[0]
	y = region[1]
	w = region[2]
	h = region[3]
	cx = x+w/2
	cy = y+h/2
	return cx, cy, w, h

def show_frame(frame, bbox):
	frame_adjusted = np.ndarray(shape=np.shape(frame), dtype=np.dtype(np.uint8))
	frame_adjusted[:,:,:] = frame[:,:,2::-1]
	x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
	cv2.rectangle(frame_adjusted, (x, y), (x+w,y+h), (0,0,255), 4, 8, 0)
	cv2.imshow('image',frame_adjusted)
	cv2.waitKey(1) # Introduces a delay of atleast 1ms.... which is around 18ms on Windows

def server_process():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("",PORT))
	s.listen(1000)
	X = 245.
	Y = 165.
	W = 150.
	H = 150.
	with tf.Session() as sess:
		model = load_model('modelv1.07-0.96.hdf5')
		dense = model.get_layer('dense').get_weights()
		weights = dense[0].T
		## This is to initialized CUDA and the required packages 
		pil_image= load_img(frame)
		img = preprocess_image(pil_image)
		result = sess.run(model.outputs, feed_dict={model.input.name:img})
		
		while True:
			print("Server is ready to accept connection")
			conn, addr = s.accept()

			tmp_data = conn.recv(4096)
			received_data = b""
			while tmp_data != b"":
				received_data += tmp_data
				tmp_data = conn.recv(4096)
			conn.shutdown(socket.SHUT_RD)

			if len(received_data) >= 8:
				mode = received_data[0:8].decode()
				with open(frame,"wb") as f:
					f.write(received_data[8:])

				pil_image= load_img(frame)
				img = preprocess_image(pil_image)
				if mode == 'classify':
					#show_frame(np.asarray(pil_image), [900.,900.,900.,900.])
					result = -1
					result = sess.run(model.outputs, feed_dict={model.input.name:img})
					conn.sendall(str(int(np.argmax(result))).encode().rjust(4,b'0'))
				elif mode == 'tracking':
					result, xmin, ymin, xmax, ymax = -1, -1, -1, -1, -1
					[result,out_relu] = sess.run([model.outputs,model.get_layer('out_relu').output], feed_dict={model.input.name:img})
					category = np.argmax(result)
					print('Category: ',category)
					if category == 1:
						kernels = out_relu.reshape(7,7,1280)
						cam = np.dot(kernels,weights[result[0].argmax()])
						cam_array = array_to_img(cam.reshape(7,7,1))
						cam_array = cam_array.resize((224,224), Image.ANTIALIAS)
						box = img_to_array(cam_array).reshape(224,224)
						cam_threshold = (box > box.max()*.8) *1
						cam_threshold_adjusted = np.ndarray(shape=np.shape(cam_threshold), dtype=np.dtype(np.uint8))
						cam_threshold_adjusted[:,:] = np.asarray(cam_threshold)*255
						contours, _ = cv2.findContours(cam_threshold_adjusted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
						contours = np.array(contours)
						max_area = [0,0] # contours index and area
						for index, contour in enumerate(contours):
							if(max_area[1]< len(contour)):
								max_area = [index,len(contour)]	
						contours_adjusted = contours[max_area[0]].squeeze(axis=1).T
						xmin = contours_adjusted[0].min() * (640./224.)
						ymin = contours_adjusted[1].min() * (480./224.)
						xmax = contours_adjusted[0].max() * (640./224.)
						ymax = contours_adjusted[1].max() * (480./224.)
						#ymin_temp = ymin + (ymax - ymin)/2. - H/2.
						#xmin_temp = xmin + (xmax - xmin)/2. - W/2.
						#show_frame(np.asarray(pil_image), [xmin_temp, ymin_temp, W, H])
						#print("xmin: ",xmin,"\t ymin: ",ymin, 'xmax: ',xmax,'\t ymax: ',ymax)
						#print(str(int(xmin)).encode().rjust(3,b'0')+str(int(ymin)).encode().rjust(3,b'0')+str(int(xmax)).encode().rjust(3,b'0')+str(int(ymax)).encode().rjust(3,b'0'))
					conn.sendall(str(int(category)).encode().rjust(4,b'0')+
							str(int(xmin)).encode().rjust(3,b'0')+
							str(int(ymin)).encode().rjust(3,b'0')+
							str(int(xmax)).encode().rjust(3,b'0')+
							str(int(ymax)).encode().rjust(3,b'0'))

			conn.close()
		s.close()

if __name__ == '__main__':
	sys.exit(server_process())
