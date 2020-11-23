from __future__ import division
import sys
import os
import msvcrt, time
import numpy as np
import cv2
from PIL import Image
from src.region_to_bbox import region_to_bbox
from src.visualization import show_frame
from cozmo.util import degrees, distance_mm, speed_mmps
result =''
import cozmo
from tensorflow.keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import math

global directory
directory = os.path.join(os.getcwd(), "test")
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array,array_to_img
from tensorflow.keras.models import load_model

def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	# Pre-process input data
	img = preprocess_input(img)
	return img

def cozmo_program(robot: cozmo.robot.Robot):
	
	global angle
	angle = 25.
	robot.set_head_angle(degrees(angle)).wait_for_completed()
	robot.set_lift_height(0.0).wait_for_completed()
	robot.camera.image_stream_enabled = True
	robot.camera.color_image_enabled = True
	robot.camera.enable_auto_exposure = True
	
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	
	frame = os.path.join(directory, "current.jpeg")

	print("Starting Tensorflow...")
	
	with tf.Session() as sess:
		print("Session successfully started")
		model = load_model('modelv1.07-0.96.hdf5')		
		while True:
			global X, Y, W, H
			global result
			X = 245.
			Y = 165.
			W = 150.
			H = 150.
			
			gt = [X, Y, W, H]
			pos_x, pos_y, target_w, target_h = region_to_bbox(gt)
			frame = os.path.join(directory, "current.jpeg")
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
					pilImage.resize((640,480), Image.ANTIALIAS).save(os.path.join(directory, "current.jpeg"), "JPEG") 
				show_frame(np.asarray(Image.open(frame)), [900.,900.,900.,900.], 1)
				img = load_image(frame)
				[result,out_relu,global_average_pooling2d] = sess.run([model.outputs,model.get_layer('out_relu').output\
										   ,model.get_layer('global_average_pooling2d').output ], feed_dict={model.input.name:img})
				next_state = np.argmax(result)
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
							dense = model.get_layer('dense').get_weights()
							weights = dense[0].T
							
							testing_counter = 0
							detected_centroid = 0
							xmin_avg = 0
							xmax_avg = 0
							ymin_avg = 0
							ymax_avg = 0
							frame_average = 2
							frame_count = 0
							while True:
								latest_img = robot.world.latest_image
								if latest_img is not None:
									pilImage = latest_img.raw_image
									pilImage.resize((640,480), Image.ANTIALIAS).save(os.path.join(directory, "current.jpeg"), "JPEG")
								img = load_image(frame)
								[result,out_relu,global_average_pooling2d] = sess.run([model.outputs,model.get_layer('out_relu').output\
														   ,model.get_layer('global_average_pooling2d').output ], feed_dict={model.input.name:img})
								
								kernels = out_relu.reshape(7,7,1280)
								final = np.dot(kernels,weights[result[0].argmax()])
								final1 = array_to_img(final.reshape(7,7,1))
								final1 = final1.resize((224,224), Image.ANTIALIAS)
								box = img_to_array(final1).reshape(224,224)
								#box = cv2.blur(box,(30,30))
								temp = (box > box.max()*.8) *1 
								
								temp_adjusted = np.ndarray(shape=np.shape(temp), dtype=np.dtype(np.uint8))
								temp_adjusted[:,:] = np.asarray(temp)*255
								contours, hierarchy = cv2.findContours(temp_adjusted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
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
																
								if result[0].argmax() == 1:
									
									# Frame smoothing
									frame_count = frame_count + 1
									xmin_avg = xmin_avg + xmin
									xmax_avg = xmax_avg + xmax
									ymin_avg = ymin_avg + ymin
									ymax_avg = ymax_avg + ymax
									
									if frame_count % frame_average == 0:
										frame_count = 0
										xmin_avg = xmin_avg/frame_average
										xmax_avg = xmax_avg/frame_average
										ymin_avg = ymin_avg/frame_average
										ymax_avg = ymax_avg/frame_average
										
										print(xmin_avg, end=",")
										print(ymin_avg, end=",")
										print(xmax_avg, end=",")
										print(ymax_avg, end="\n")
										ymin_avg = ymin_avg + (ymax_avg - ymin_avg)/2. - H/2.
										xmin_avg = xmin_avg + (xmax_avg - xmin_avg)/2. - W/2.
										print("150: ",xmin_avg, end=",")
										print("150: ",ymin_avg, end="\n")
										gt = [xmin_avg, ymin_avg, W, H]
										xmin_avg = 0
										xmax_avg = 0
										ymin_avg = 0
										ymax_avg = 0
										
										pos_x, pos_y, target_w, target_h = region_to_bbox(gt)
										bboxes = np.zeros((1, 4))
										#bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
										bboxes[0,:] = pos_x-W/2, pos_y-H/2, W, H
										print(len(contours))
										testing_counter = testing_counter + 1
										print("Testing_counter: ",testing_counter)
										show_frame(np.asarray(Image.open(frame)), gt, 1)
										print("Cat is detected")								
									
										print("Starting the tracker ...")
										if (bboxes[0,1] + bboxes[0,3]/2) < (Y + H/2 - 40):
											print("Command: Raise the head")
											angle = angle + 0.5
											if angle > 44.5:
												angle = 44.5
										elif (bboxes[0,1] + bboxes[0,3]/2) > (Y + H/2 + 40):
											print("Command: Lower the head")
											angle = angle - 0.5
											if angle < 0:
												angle = 0
										else:
											pass
										
										set_head_angle_action = robot.set_head_angle(degrees(angle), max_speed=20, in_parallel=True)
										
										if straight(bboxes[0,:])[0] != 0 and turn(bboxes[0,:])[0] != 0:
											robot.drive_wheel_motors(straight(bboxes[0,:])[0] + turn(bboxes[0,:])[0], straight(bboxes[0,:])[1] + turn(bboxes[0,:])[1])
											detected_centroid = 0
										elif straight(bboxes[0,:])[0] == 0 and turn(bboxes[0,:])[0] == 0:
											robot.stop_all_motors()
											detected_centroid = detected_centroid + 1
										elif straight(bboxes[0,:])[0] == 0:
											robot.drive_wheel_motors(turn(bboxes[0,:])[0], turn(bboxes[0,:])[1])
											detected_centroid = 0
										elif turn(bboxes[0,:])[0] == 0:
											robot.drive_wheel_motors(straight(bboxes[0,:])[0], straight(bboxes[0,:])[1])
											detected_centroid = 0
										else:
											robot.stop_all_motors()
											detected_centroid = detected_centroid + 1
										
										if detected_centroid > 20//frame_average:
											detected_centroid = 0
											print("Reached a stable state.........\t\t\t\t\t\t\t\t STABLE")
											
											# Go near the object
											
											set_head_angle_action.wait_for_completed()
											robot.abort_all_actions(log_abort_messages=True)
											robot.wait_for_all_actions_completed()
											robot.set_head_angle(degrees(0.5)).wait_for_completed()
											print("Robot's head angle: ",robot.head_angle)
											target_frame_count = 1
											while True:
												latest_img = None
												while latest_img is None:
													latest_img = robot.world.latest_image
												target_frame1 = latest_img.raw_image
												target_frame1 = target_frame1.resize((640,480), Image.ANTIALIAS)
												#target_frame1 = target_frame1.convert('L')
												target_frame1 = np.asarray(target_frame1)
												#orb1 = cv2.ORB_create(500)
												#kp1 = orb1.detect(target_frame1,None)
												#kp1, des1 = orb1.compute(target_frame1, kp1)
												#features_img1 = cv2.drawKeypoints(target_frame1, kp1, None, color=(255,0,0), flags=0)
												#plt.imsave("target_frame1_"+str(target_frame_count)+".jpeg",features_img1)
												plt.imsave("target_frame1_"+str(target_frame_count)+".jpeg",target_frame1)
											
												drive_straight_action = robot.drive_straight(distance=cozmo.util.distance_mm(distance_mm=10),speed=cozmo.util.speed_mmps(10), in_parallel=True)
												drive_straight_action.wait_for_completed()
												robot.set_head_angle(degrees(0.5)).wait_for_completed()
												print("Robot's head angle: ",robot.head_angle)
												latest_img = None
												while latest_img is None:
													latest_img = robot.world.latest_image
												target_frame2 = latest_img.raw_image
												target_frame2 = target_frame2.resize((640,480), Image.ANTIALIAS)
												#target_frame2 = target_frame2.convert('L')
												target_frame2 = np.asarray(target_frame2)
												#orb2 = cv2.ORB_create(500)
												#kp2 = orb2.detect(target_frame2,None)
												#kp2, des2 = orb2.compute(target_frame2, kp2)
												#features_img2 = cv2.drawKeypoints(target_frame2, kp2, None, color=(255,0,0), flags=0)
												#plt.imsave("target_frame2_"+str(target_frame_count)+".jpeg",features_img2)
												plt.imsave("target_frame2_"+str(target_frame_count)+".jpeg",target_frame2)
												target_frame_count = target_frame_count + 1
												'''
												matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
												matches = matcher.match(des1, des2, None)
												
												matches.sort(key=lambda x: x.distance, reverse=False)
												matches = matches[:10]
												imMatches = cv2.drawMatches(target_frame1, kp1, target_frame2, kp2, matches, None)
												cv2.imwrite("matches_tf1_tf2.jpg", imMatches)
												
												points1 = np.zeros((len(matches), 2), dtype=np.float32)
												points2 = np.zeros((len(matches), 2), dtype=np.float32)

												for i, match in enumerate(matches):
													points1[i, :] = kp1[match.queryIdx].pt
													points2[i, :] = kp2[match.trainIdx].pt
													print("Points1 [{}]: {}".format(i,points1[i][0]), points1[i][1],"\tPoints2: ",points2[i][0], points2[i][1]) 
												index = None
												dist1_x = []
												dist2_x = []
												for index in range(len(points1)):
													dist1_x.append((W/2.)-points1[index][0]) # Extract only the x-coordinate
													dist2_x.append((W/2.)-points2[index][0]) # Extract only the x-coordinate
																							
												fw_x = 1./((1./np.array(dist2_x)) - (1./np.array(dist1_x))) # Calculate the image plane to obj plane mapping in x direction
												
												pt1_x = []
												pt2_x = []
												for index in range(len(points1)):
													pt1_x.append(fw_x[index]/(W/2. - points1[index][0])) 
													pt2_x.append(fw_x[index]/(W/2. - points2[index][0]))
													print("Approx. distance[{}]: {}".format(index, pt1_x[index]))
												if len(pt2_x) < 10:
													break
												'''
											sys.exit(0)
											
					else:				   # Detected Dog
						dog_counter += 1
						if dog_counter >= 6:  # Transition to Dog only if Dog appeared for more than 6 times
							cat_counter = 0
							current_state = 2
				# Current State is Dog
				elif current_state == 2:
					print('\t\t\t\t\t\t\t\t\t\t\t\tDog')
					if next_state == 0:	 # Detected Background
						background_counter += 1
						if background_counter >= 6:  # Transition to Background only if Background appeared for more than 6 times
							background_counter = 0
							current_state = 0
							dog_counter = 0 
					elif next_state == 2:   # Detected Dog itself
						dog_counter +=1
						if dog_counter >= 30:
							print('Cozmo sees a Dog')
							robot.drive_wheels(-50, -50)
							time.sleep(3)
							robot.drive_wheels(70, -70)
							time.sleep(2.8)  
							robot.drive_wheels(0, 0)						
							break 
					else:				   # Detected Cat
						cat_counter += 1
						if cat_counter >= 6:  # Transition to Cat only if Cat appeared for more than 6 times
							dog_counter = 0
							current_state = 1			
		
def straight(dims):
	if ((dims[2] * dims[3]) - (W * H)) > 100 + 10*(H+W) and ((dims[2] * dims[3]) - (W * H)) < 900 + 30*(H+W):
		print("Command: Move back")
		vel = -1.5 * (((dims[2] * dims[3]) - (W * H)) % 3000)**(1.0/2)
		return vel, vel
	elif  ((dims[2] * dims[3]) - (W * H)) < -(100 + 10*(H+W)) and ((dims[2] * dims[3]) - (W * H)) > -(900 + 30*(H+W)):
		print("Command: Move straight")
		vel = 1.5 * (((W * H) - (dims[2] * dims[3]))%3000)**(1.0/2)
		return vel, vel
	elif  ((dims[2] * dims[3]) - (W * H)) >= 900 + 30*(H+W):
		print("Command: Move back")
		vel = max(-2.0 * (((dims[2] * dims[3]) - (W * H)) % 3000)**(1.0/2), -80)
		return vel, vel
	elif  ((dims[2] * dims[3]) - (W * H)) <= -(900 + 30*(H+W)):
		print("Command: Move straight")
		vel = max(2.0 * (((W * H) - (dims[2] * dims[3]))%3000)**(1.0/2), 80)
		return vel, vel
	else:
		return 0, 0
		
		
def turn(dims):
	if ((dims[0] + dims[2]/2) - (X + W/2)) > 20: # > 20
		print("Command: Turn right")
		vel = max(5 + 0.125 * ((dims[0] + dims[2]/2) - (X + W/2)), 10)
		return vel, -vel
	elif ((dims[0] + dims[2]/2) - (X + W/2)) < -20: # < -20
		print("Command: Turn left")
		vel = max(-5 + 0.125 * ((dims[0] + dims[2]/2) - (X + W/2)), -10)
		return vel, -vel
	else:
		return 0, 0
		 
		
def cozmo_run():
	cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False) 
	

if __name__ == '__main__':
	sys.exit(cozmo_run())