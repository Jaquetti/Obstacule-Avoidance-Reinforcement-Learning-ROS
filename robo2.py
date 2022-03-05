#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
from std_msgs.msg import UInt16
from std_srvs.srv import Empty
import itertools
import time
import skfuzzy.control as ctrl
import skfuzzy as fuzz
import random
import math
from math import *
#from ex2b.msg import Num
import os
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np
import tf
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

rang = 0
posi_x = 0
posi_y = 0
sum_r = 0


def back_gr():
	rospy.wait_for_service('/reset_positions')
	clear_bg = rospy.ServiceProxy('/reset_positions', Empty)
	clear_bg()

def checkCrash(lidar):
	var = 0
	if min(rang)<0.05:
		var = 1
	return var


def get_pos(data):
	global posi_x,posi_y
	global theta

	posi_x = data.pose.pose.position.x
	posi_y = data.pose.pose.position.y
	(roll, pitch, yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
	theta = yaw


def sensor_read(msgScan):
	global rang
	rang = msgScan.ranges
	MAX_LIDAR_DISTANCE = 5
	distance = []
	angle = []

	for i in range(len(msgScan.ranges)):
		angle  = math.degrees(i*msgScan.angle_increment)
		if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
			distance.append(MAX_LIDAR_DISTANCE)
		else:
			distance.append(msgScan.ranges[i])
		
	distances = np.array(distance)
	angles = np.array(angle)

	return  distances, angles

posi = 0
theta = 0

rospy.init_node('robo1')
sub = rospy.Subscriber('/robot_1/odom',Odometry, get_pos)
pub = rospy.Publisher('/robot_1/cmd_vel', Twist, queue_size = 10)
posi  = Odometry()	
rate = rospy.Rate(10)




def get_state(lidar, state_space):
	threshold  = min(lidar)+0.5 if min(lidar)<1.5 else 0
	#print(threshold)
	x1 = 0 #lado direito
	x2 = 0 #lado esquerdo
	x3 = 0 #lado esquerdo zona1
	x4 = 0 #lado esquerdo zona0
	x5 = 0 #lado direito zona0
	x6 = 0 #lado direito zona1

	if np.any(threshold>=lidar[0:135]):
		x1 = 1
	if np.any(threshold>=lidar[135:270]):
		x2 = 1	

	if np.any(threshold>=lidar[195:270]):
		x3 = 1

	if np.any(threshold>= lidar[135:195]):
		x4 = 1

	if np.any(threshold>= lidar[75:135]):
		x5 = 1

	if np.any(threshold>=lidar[0:75]):
		x6 = 1
	
	state1 = (x1,x2,x3,x4,x5,x6)
	
	state = state_space.index(state1) 
	
	return state

def create_state_space():
	a = [1,0]
	b = [1,0]
	c = [1,0]
	d = [1,0]
	e = [1,0]
	f = [1,0]

	state_space = list(itertools.product(a, b,c,d,e,f))
	state_space.reverse()

	return state_space

def create_state_space():
	a = [1,0]
	b = [1,0]
	c = [1,0]
	d = [1,0]
	e = [1,0]
	f = [1,0]

	state_space = list(itertools.product(a, b,c,d,e,f))
	state_space.reverse()

	return state_space

def create_action_space():
	action_space = np.array([0,1,2,3])
	return action_space

def take_action(a):
	vel = Twist()
	
	if a ==0:
		vel.linear.x = 0.2
		vel.linear.y = 0
		vel.linear.z = 0	
		vel.angular.x = 0
		vel.angular.y = 0
		vel.angular.z = 0

	if a ==1:
		vel.linear.x = 0.2
		vel.linear.y = 0
		vel.linear.z = 0	
		vel.angular.x = 0
		vel.angular.y = 0
		vel.angular.z = 0.6

	if a ==2:
		vel.linear.x = 0.2
		vel.linear.y = 0
		vel.linear.z = 0	
		vel.angular.x = 0
		vel.angular.y = 0
		vel.angular.z = -0.6

	if a ==3:
		vel.linear.x = 0
		vel.linear.y = 0
		vel.linear.z = 0	
		vel.angular.x = 0
		vel.angular.y = 0
		vel.angular.z = 0			

	pub.publish(vel)
	rate.sleep()

def stop_robot():
	vel = Twist()
	vel.linear.x = 0
	vel.linear.y = 0
	vel.linear.z = 0	
	vel.angular.x = 0
	vel.angular.y = 0
	vel.angular.z = 0
	pub.publish(vel)
	rate.sleep()


def checkCrash(lidar):
	var = 0
	if min(lidar)<0.05:
		var = 1
	return var

def get_reward(action, prev_action, lidar, prev_lidar, crash, prev_distance, distance):
	if crash ==1:
		reward = -100
	else:
		if action==0:
			r_action = 0.2
		else:
			r_action = -0.1

		if min(prev_lidar)<min(lidar):
			r_obs = 0.2
		else:
			r_obs = -0.2

		if ( prev_action == 1 and action == 2 ) or ( prev_action == 2 and action == 1 ):
			r_change = -0.8
		else:
			r_change = 0.0
		
		if distance<prev_distance:
			r_dis = 0.5
		else:
			r_dis = -0.5
		
		if ( prev_action == 3 and action == 3 ):
			r_stop = -5
		else:
			r_stop = 0.0

		# Cumulative reward
		reward = r_action + r_obs + r_change + r_stop + r_dis

	return reward


def pub_vel(out1,out2):


	vel = Twist()
	vel.linear.x = out1
	vel.linear.y = 0
	vel.linear.z = 0	
	vel.angular.x = 0
	vel.angular.y = 0
	vel.angular.z = out2


	pub.publish(vel)
	rate.sleep()

def stop_robot():
	vel = Twist()
	vel.linear.x = 0
	vel.linear.y = 0
	vel.linear.z = 0	
	vel.angular.x = 0
	vel.angular.y = 0
	vel.angular.z = 0
	pub.publish(vel)
	rate.sleep()

set_point_x = 0
i=0
state_space = create_state_space()
action_space = create_action_space()
acao_tomada = {0:'para frente', 1:'para a esquerda',2:'para a direita', 3:'parar'}

cont_re = 0

pontos_pegar = [[-7,-1],[-7,6],[-3.38,6.77], [3.73,7],[-7.23,-8.25]]
pontos_soltar = [[4.69,3.64],[4.5,3.5],[4.5,2],[4.5,0],[4.5,-1.52]]



epochs = 500
num_max_steps = 3000
epsilon = 0
gama = 0.99
lr = 0.1
exp = 1
max_exp = 1
min_exp = 0.01
exp_decay = 0.001
#alpha = 0.8
gamma = 0.999
temp = 2
dist = 0


state_space = create_state_space()
action_space = create_action_space()


def policy(Q,state, t):
    p = np.array([Q[(state,x)]/t for x in range(len(create_action_space()))])
    prob_actions = np.exp(p) / np.sum(np.exp(p))
    cumulative_probability = 0.0
    choice = random.uniform(0,1)
    for a,pr in enumerate(prob_actions):
        cumulative_probability += pr
        if cumulative_probability > choice:
            return a

try:
	Q  = np.load('Q1.npy')
	print('Peso carregado')

except:
	print('Nao existe peso')

	Q = np.zeros((len(state_space),len(action_space)))



back_gr()


def robotFeedbackControl(x, y, theta, x_goal, y_goal, theta_goal):
	K_RO = 2
	K_ALPHA = 15
	K_BETA = -3
	V_CONST = 0.8

	
	if theta_goal >= pi:
		theta_goal_norm = theta_goal - 2 * pi
	else:
		theta_goal_norm = theta_goal

	ro = sqrt(pow((x_goal-x),2) + pow((y_goal -y) ,2) )
	lamda = atan2(y_goal- y,x_goal- x)

	alpha = (lamda -  theta + pi) % (2 * pi) - pi
	beta = (theta_goal - lamda + pi) % (2 * pi) - pi


	if ro < 0.5:
		status = 1
		v = 0
		w = 0
		v_scal = 0
		w_scal = 0
	else:
		status = 0
		v = K_RO * ro
		w = K_ALPHA * alpha + K_BETA * beta
		v_scal = v / abs(v) * V_CONST
		w_scal = w / abs(v) * V_CONST

	velMsg = pub_vel(v_scal, w_scal)
		

	return status

fab_info1 = []
fab_info2 = []
esteira1 = 0
esteira2 = 0
esteira3 = 0
esteira4 = 0
esteira5 = 0
trans1 = 0
trans2 = 0
trans3 = 0
trans4 = 0


def get_data1(msg):
	global esteira1
	esteira1  = msg.data

def get_data2(msg):
	global esteira2
	esteira2  = msg.data

def get_data3(msg):
	global esteira3
	esteira3  = msg.data

def get_data4(msg):
	global esteira4
	esteira4  = msg.data


def get_data5(msg):
	global esteira5
	esteira5  = msg.data

def get_data6(msg):
	global trans1
	trans1  = msg.data

def get_data7(msg):
	global trans2
	trans2  = msg.data

def get_data8(msg):
	global trans3
	trans3  = msg.data

def get_data9(msg):
	global trans4
	trans4  = msg.data



pub_fab1 = rospy.Publisher('/esteira1', UInt16,queue_size = 10)
pub_fab2 = rospy.Publisher('/esteira2', UInt16,queue_size = 10)
pub_fab3 = rospy.Publisher('/esteira3', UInt16,queue_size = 10)
pub_fab4 = rospy.Publisher('/esteira4', UInt16,queue_size = 10)
pub_fab5 = rospy.Publisher('/esteira5', UInt16,queue_size = 10)
pub_fab6 = rospy.Publisher('/trans1', UInt16,queue_size = 10)
pub_fab7 = rospy.Publisher('/trans2', UInt16,queue_size = 10)
pub_fab8 = rospy.Publisher('/trans3', UInt16,queue_size = 10)
pub_fab9 = rospy.Publisher('/trans4', UInt16,queue_size = 10)

sub_fab1 = rospy.Subscriber('/esteira1',UInt16, get_data1)
sub_fab2 = rospy.Subscriber('/esteira2',UInt16, get_data2)
sub_fab3 = rospy.Subscriber('/esteira3',UInt16, get_data3)
sub_fab4 = rospy.Subscriber('/esteira4',UInt16, get_data4)
sub_fab5 = rospy.Subscriber('/esteira5',UInt16, get_data5)
sub_fab6 = rospy.Subscriber('/trans1',UInt16, get_data6)
sub_fab7 = rospy.Subscriber('/trans2',UInt16, get_data7)
sub_fab8 = rospy.Subscriber('/trans3',UInt16, get_data8)
sub_fab9 = rospy.Subscriber('/trans4',UInt16, get_data9)


s = 0

def pub_point1(i):
	if i ==0:
		pub_fab1.publish(1)
	if i ==1:
		
		pub_fab2.publish(1)
	if i ==2:
		
		pub_fab3.publish(1)
	if i ==3:
	
		pub_fab4.publish(1)
	if i ==4:
		
		pub_fab5.publish(1)

	rate.sleep()



def pub_point2(i):
	if i ==0:
		pub_fab1.publish(0)
	if i ==1:
		
		pub_fab2.publish(0)
	if i ==2:
		
		pub_fab3.publish(0)
	if i ==3:
	
		pub_fab4.publish(0)
	if i ==4:
		
		pub_fab5.publish(0)

	rate.sleep()


		
def pub_point3(i):
	#fab = Num()
	if i ==0:
		pub_fab6.publish(1)
	if i ==1:
		pub_fab7.publish(1)
	if i ==2:
		pub_fab8.publish(1)
	if i ==3:
		pub_fab9.publish(1)

	rate.sleep()

def pub_point4(i):
	#fab = Num()
	if i ==0:
		pub_fab6.publish(0)
	if i ==1:
		pub_fab7.publish(0)
	if i ==2:
		pub_fab8.publish(0)
	if i ==3:
		pub_fab9.publish(0)
		
	rate.sleep()

pontos_pegar = [[-7,-1],[-7,6],[-3.38,6.77], [3.73,7],[-7.23,-8.25]]
pontos_soltar = [[4.5,3.5],[4.5,2],[4.5,0],[4.5,-1.52]]

flag1 = 0
flag2 = 0

d1 = []
m1 = []


d2 = []
m2 = []
	

while not rospy.is_shutdown():
	while True:
		i = random.randint(0, len(pontos_pegar)-1)
		set_point_x = float(pontos_pegar[i][0])
		set_point_y = float(pontos_pegar[i][1])
		dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
		ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x)
		try:
			Q  = np.load('Q1.npy')
			print('pegando a carga')
		except:
			pass

		while True:
			fab_info1  = [esteira1,esteira2,esteira3,esteira4,esteira5]
			posi  = Odometry()
			dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
			ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x) - theta
			msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
			(lidar, angles) = sensor_read(msgScan)

			if dist<2 and fab_info1[i] ==1:
				flag1 = 1
			else:
				flag1 = 0

			d1.append(dist)

			if len(d1) ==4 and flag1 ==0:
				if sum(d1)/4 == d1[0]:
					back_gr()
				else:
					pass

				d1 = []


			if min(lidar)<0.5 and flag1==0:
				msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
				(lidar, angles) = sensor_read(msgScan)
				prev_lidar = lidar
				prev_action = 0
				prev_distance = dist
				state = get_state(lidar,state_space)
				crash = 0
				sum_r = 0
				steps = 0
				cont_re = 0

				while min(lidar)<0.5:
					msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
					(lidar, angles) = sensor_read(msgScan)
					a = policy(Q, state, temp)
					reward = take_action(a)
					#m1.append(dist)
					if a!=3:
						m1.append(dist)


					msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
					(lidar, angles) = sensor_read(msgScan)
					new_state = get_state(lidar,state_space)
					crash = checkCrash(lidar)
					dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
					ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x) - theta

					
					r = get_reward(a, prev_action, lidar, prev_lidar, crash, prev_distance, dist)
					
					if len(m1) ==4:
						if sum(m1)/4 == m1[0]:
							r = -100
						else:
							pass

						m1 = []


					Q[state,a] = Q[state,a]*(1-lr) + lr*(r+gama*np.max(Q[new_state,:]))

					state = new_state
					prev_lidar = lidar
					prev_action = a
					prev_distance = dist

					if crash==1 or r==-100:
						if temp > 1.0:
							temp -= 0.01

						
						back_gr()
		
						break

					np.save('Q1.npy', Q)


			else:
				
				s = robotFeedbackControl(posi_x, posi_y, theta, set_point_x, set_point_y, ang)
				

			if fab_info1[i] ==1 and dist <= 1.4:
				while fab_info1[i] ==1:
					fab_info1  = [esteira1,esteira2,esteira3,esteira4,esteira5]
					stop_robot()

			if s==1:
				break

		tempo_agora = time.time()
		l = 0
		while tempo_agora+4>time.time():
			if l ==0:
				l=1
				pub_point1(i)

		pub_point2(i)

		i = random.randint(0, len(pontos_soltar)-1)
		
		set_point_x = float(pontos_soltar[i][0])
		set_point_y = float(pontos_soltar[i][1])
		dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
		ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x)

		print('trazendo carga')

		while True:
			fab_info2  = [trans1,trans2,trans3,trans4]
			posi  = Odometry()
			dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
			ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x) - theta
			msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
			(lidar, angles) = sensor_read(msgScan)
			


			if dist<2 and fab_info2[i] ==1:
				flag2 = 1
			else:
				flag2 = 0


			d2.append(dist)

			if len(d2) ==4 and flag2==0:
				if sum(d2)/4 == d2[0]:
					back_gr()
				else:
					pass

				d2 = []


			if min(lidar)<0.5 and flag2==0:
				
				msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
				(lidar, angles) = sensor_read(msgScan)
				prev_lidar = lidar
				prev_action = 0
				prev_distance = dist
				state = get_state(lidar,state_space)
				crash = 0
				sum_r = 0
				steps = 0
				cont_re = 0

				while min(lidar)<0.5:
					msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
					(lidar, angles) = sensor_read(msgScan)
					a = policy(Q, state, temp)
					reward = take_action(a)
					msgScan = rospy.wait_for_message('/robot_1/base_scan', LaserScan)
					(lidar, angles) = sensor_read(msgScan)
					new_state = get_state(lidar,state_space)
					crash = checkCrash(lidar)
					dist = math.sqrt((set_point_x-posi_x)**2+(set_point_y-posi_y)**2)
					ang =  math.atan2(set_point_y-posi_y,set_point_x-posi_x) - theta

					
					r = get_reward(a, prev_action, lidar, prev_lidar, crash, prev_distance, dist)
					

					#m2.append(dist)
					if a!=3:
						m2.append(dist)
					if len(m2) ==4:
						if sum(m2)/4 == m2[0]:
							r = -100
						else:
							pass

						m2 = []

					Q[state,a] = Q[state,a]*(1-lr) + lr*(r+gama*np.max(Q[new_state,:]))

					state = new_state
					prev_lidar = lidar
					prev_action = a
					prev_distance = dist


					if crash==1 or r==-100:
						if temp > 1.0:
							temp -= 0.01
						back_gr()
		
						break

					np.save('Q1.npy', Q)


			else:
				s = robotFeedbackControl(posi_x, posi_y, theta, set_point_x, set_point_y, ang)
				
			if fab_info2[i] ==1 and dist <= 1.4:
				while fab_info2[i] ==1:
					fab_info2  = [trans1,trans2,trans3,trans4]
					stop_robot()

			if s==1:
				break

		tempo_agora = time.time()
		l = 0
		while tempo_agora+4>time.time():
			if l ==0:
				l=1
				pub_point3(i)

		pub_point4(i)