from __future__ import division
import sys
import os
import time
import numpy as np
import cv2
import cozmo
import socket

from PIL import Image
from cozmo.util import degrees

global frame 
frame = "current.jpg"
HOST1, PORT1 = '192.168.1.105',2000
HOST2, PORT2 = '192.168.1.51',2000
TIMEOUT = 2.0 # Range from 0.2 to 2.0 depending on the network stability and wired or wireless scenario
HOST, PORT = HOST1, PORT1
#Wired Server 
#   (ASUS) Address - '192.168.137.149',2000 	
#Wireless Server 
#   (ASUS) Address - '192.168.1.105',2000
#   (Jetson) Address - '192.168.1.51',2000

def region_to_bbox(region):
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

def get_classification_result(file_path):
	result = -1
	global HOST, PORT, HOST1, PORT1, HOST2, PORT2, TIMEOUT
	try:
		with open(file_path, 'rb') as f:
			senddata = f.read()
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.settimeout(TIMEOUT)
			s.connect((HOST,PORT))
			s.sendall(b'classify') # Send 'classify' as identifier for classification
			s.sendall(senddata)
		s.shutdown(socket.SHUT_WR)

		tmp_data = s.recv(4096)
		received_data = b""
		while tmp_data != b"":
			received_data += tmp_data
			tmp_data = s.recv(4096)
		s.close()

		if len(received_data) == 4: # Otherwise the received data is incomplete
			result = int(received_data[0:4].decode())
	except Exception as e:
		print(e)
		print('Lost connection to server: ',HOST,':',PORT)
		if HOST == HOST1:
			HOST = HOST2
			PORT = PORT2
		elif HOST == HOST2:
			HOST = HOST1
			PORT = PORT1
		print('Trying to connect to a different server: ',HOST,':',PORT)
	finally:
		s.close()
	return result

def get_tracking_result(file_path):
	classification_result, pos_x, pos_y, w, h = -1, -1, -1, -1, -1
	global HOST, PORT, HOST1, PORT1, HOST2, PORT2, TIMEOUT
	try:
		with open(file_path, 'rb') as f:
			senddata = f.read()
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.settimeout(TIMEOUT)
			s.connect((HOST,PORT))
			s.sendall(b'tracking') # Send 'tracking' as identifier for tracking
			s.sendall(senddata)
		s.shutdown(socket.SHUT_WR)

		tmp_data = s.recv(4096)
		received_data = b""
		while tmp_data != b"":
			received_data += tmp_data
			tmp_data = s.recv(4096)
		s.close()

		if len(received_data) == 16: # Otherwise the received data is incomplete
			classification_result = int(received_data[0:4].decode())
			if classification_result == 1:
				pos_x = int(received_data[4:7].decode())
				pos_y = int(received_data[7:10].decode())
				w = int(received_data[10:13].decode())
				h = int(received_data[13:16].decode())
	except Exception as e:
		print(e)
		print('Lost connection to server: ',HOST,':',PORT)
		if HOST == HOST1:
			HOST = HOST2
			PORT = PORT2
		elif HOST == HOST2:
			HOST = HOST1
			PORT = PORT1
		print('Trying to connect to a different server: ',HOST,':',PORT)
	finally:
		s.close()
	return classification_result, pos_x, pos_y, w, h

def cozmo_program(robot: cozmo.robot.Robot):

	angle = 25.
	robot.set_head_angle(degrees(angle)).wait_for_completed()
	robot.set_lift_height(0.0).wait_for_completed()
	robot.camera.image_stream_enabled = True
	robot.camera.color_image_enabled = True
	robot.camera.enable_auto_exposure = False
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	while True:
		global X, Y, W, H
		X = 245.
		Y = 165.
		W = 150.
		H = 150.
		pos_x, pos_y, target_w, target_h = region_to_bbox([X, Y, W, H])
		result = 0
		dog_counter = 0
		cat_counter = 0
		background_counter = 0
		next_state = 0
		current_state = 0 #Background: 0, Cat:1, Dog:2
		while True:
			latest_img = robot.world.latest_image
			if latest_img is not None:
				pilImage = latest_img.raw_image
				pilImage.resize((640,480), Image.ANTIALIAS).save(frame, "JPEG") # Resize the image (320x240) to (640x480) for frame visualization

			pil_image= Image.open(frame)
			show_frame(np.asarray(pil_image), [900.,900.,900.,900.])

			next_state = get_classification_result(frame)
			print('Arg max: ',next_state)
			# Initial Current State is Background
			if current_state == 0: 
				print('Background')
				if next_state == 1: # Detected a Cat
					current_state = 1   # Transition to Cat State
					background_counter = 0
					cat_counter = 1
					dog_counter = 0
				elif next_state == 2: # Detected a Dog
					current_state = 2   # Transition to Dog state
					background_counter = 0
					cat_counter = 0
					dog_counter = 1
			# Current State is Cat
			elif current_state == 1: 
				print('\t\t\t\t\t\tCat')
				if next_state == 0:   # Detected Background
					background_counter += 1
					if background_counter >= 6:  # Transition to Background only if Background appeared for more than 6 times
						background_counter = 0
						current_state = 0
						cat_counter = 0
				elif next_state == 1: # Detected Cat itself
					cat_counter +=1
					if cat_counter >= 30:
						print('Cozmo sees a cat')
						# Tracking mode has started

						detected_centroid = 0
						noncat_counter = 0
						frame_count = 1
						frame_average = 1
						xmin_avg = np.zeros(frame_average)
						xmax_avg = np.zeros(frame_average)
						ymin_avg = np.zeros(frame_average)
						ymax_avg = np.zeros(frame_average)
						while True:
							latest_img = robot.world.latest_image
							if latest_img is not None:
								pilImage = latest_img.raw_image
								pilImage.resize((640,480), Image.ANTIALIAS).save(frame, "JPEG")
							pil_image= Image.open(frame)

							next_state, xmin, ymin, xmax, ymax = get_tracking_result(frame)
							print('Arg max: ',next_state)

							if next_state == 1:
								noncat_counter = 0
								# Frame smoothing
								xmin_avg = np.append(xmin_avg[1:], xmin)
								xmax_avg = np.append(xmax_avg[1:], xmax)
								ymin_avg = np.append(ymin_avg[1:], ymin)
								ymax_avg = np.append(ymax_avg[1:], ymax)
								
								if frame_count % (frame_average) == 0:
									xmin = xmin_avg.mean()
									xmax = xmax_avg.mean()
									ymin = ymin_avg.mean()
									ymax = ymax_avg.mean()
									ymin = ymin + (ymax - ymin)/2. - H/2.
									xmin = xmin + (xmax - xmin)/2. - W/2.
									show_frame(np.asarray(pil_image), [xmin, ymin, W, H])
								else:
									frame_count = frame_count + 1
							else:	# Detected Non cat
								noncat_counter += 1
								if noncat_counter >= 6:	# Transition to Non Cat only if Background appeared for more than 15 times
									noncat_counter = 0
									current_state = 0
									cat_counter = 0	
									break
									
				else:				# Detected Dog
					dog_counter += 1
					if dog_counter >= 6:  # Transition to Dog only if Dog appeared for more than 6 times
						cat_counter = 0
						current_state = 2
			# Current State is Dog
			elif current_state == 2:
				print('\t\t\t\t\t\t\t\t\t\t\t\tDog')
				if next_state == 0:		# Detected Background
					background_counter += 1
					if background_counter >= 6:		# Transition to Background only if Background appeared for more than 6 times
						background_counter = 0
						current_state = 0
						dog_counter = 0 
				elif next_state == 2:	# Detected Dog itself
					dog_counter +=1
					if dog_counter >= 30:
						print('Cozmo sees a Dog')
						robot.drive_wheels(-50, -50)
						time.sleep(3)
						robot.drive_wheels(70, -70)
						time.sleep(2.8)  
						robot.drive_wheels(0, 0)
						break 
				else:				# Detected Cat
					cat_counter += 1
					if cat_counter >= 6:  # Transition to Cat only if Cat appeared for more than 6 times
						dog_counter = 0
						current_state = 1

def cozmo_run():
	cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False) 

if __name__ == '__main__':
	sys.exit(cozmo_run())