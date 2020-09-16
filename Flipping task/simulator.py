# -*- coding: utf-8 -*-

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')

import sys
import time
import numpy as np
from math import sin, cos
import math as ms


def start_simulator(port_num):
    print ('Program started')
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', port_num, True, True, 5000, 5) # set the port number to enable continuous remote API
    vrep.simxSynchronous(clientID,True)
    if clientID != -1:
        print ('Connected to remote API server')
    else:
        print ('Failed connecting to remote API server. ' + str(clientID) + str(port_num))
        sys.exit('Program Ended')
    return clientID

def error_check(clientID, res):
    if res != vrep.simx_return_ok:
        print ('Failed to get sensor Handler')
        vrep.simxFinish(clientID)
        sys.exit('Program ended')

def wait_until_simulator_started(clientID):
    vrep.simxSynchronous(clientID,True)
    while True:
        try:
            if vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait) == vrep.simx_return_ok:
#                print('restart succeessfully')
                break
        except KeyboardInterrupt:
            sys.exit('Program Ended')

def starting_streaming_buffer_mode(clientID, handle, function_name):
    if function_name == 'simxGetObjectPosition':
        vrep.simxGetObjectPosition(clientID, handle, -1, vrep.simx_opmode_streaming)
        while True:
            returnCode,data=vrep.simxGetObjectPosition(clientID, handle, -1, vrep.simx_opmode_buffer) 
            if returnCode==vrep.simx_return_ok:
                break
            time.sleep(0.001)
#            print('starting streaming/buffer mode')
    elif function_name == 'simxGetObjectOrientation':     
        vrep.simxGetObjectOrientation(clientID, handle, -1, vrep.simx_opmode_streaming)
        while True:
            returnCode,data=vrep.simxGetObjectOrientation(clientID, handle, -1, vrep.simx_opmode_buffer) 
            if returnCode==vrep.simx_return_ok:
                break
            time.sleep(0.001)
#    print('starting streaming/buffer mode')
            
def episodic_spillage_detection(clientID, cup_handle, particle_handle, distance_threshold):
    total_spilled_amount = 0 
    returnCode, cup_position = vrep.simxGetObjectPosition(clientID, cup_handle, -1, vrep.simx_opmode_blocking)
    if returnCode != 0:
        print('Can not find Cup positions!')            
    # ! To check, if the particle is already out of border, using object count.
    for i in range(len(particle_handle)):
        returnCode, particle_position = vrep.simxGetObjectPosition(clientID, particle_handle[i], -1, vrep.simx_opmode_buffer)    #vrep.simx_opmode_blocking
        if returnCode!= 0 :
            print('cannot get particle position.')
            total_spilled_amount += 1
        elif np.linalg.norm(np.array(particle_position) - np.array(cup_position)) > distance_threshold:
            total_spilled_amount += 1
#    print(total_spilled_amount)
    return total_spilled_amount


#def step_check_routine(clientID, cup_handle, end_effector_handle, end_effector_posistion_last, end_effector_orient_vector_last, particle_handle, distance_threshold):
#    '''
#    end_effector_orientation_vector : the normal vector of the end_effector in 3d
#    '''
#    step_spillage = 0
#    returnCode, cup_position = vrep.simxGetObjectPosition(clientID, cup_handle, -1, vrep.simx_opmode_buffer)
#    returnCode, end_effector_orientation = vrep.simxGetObjectOrientation(clientID, end_effector_handle, -1, vrep.simx_opmode_buffer)
#    returnCode, end_effector_position_new = vrep.simxGetObjectPosition(clientID, end_effector_handle, -1, vrep.simx_opmode_buffer)    
#    # Post-processing
#    end_effector_position_difference = np.linalg.norm((np.array(end_effector_position_new) - np.array(end_effector_posistion_last)), ord=2)
#    R = vrepEulerRotation(end_effector_orientation)
#    end_effector_orientation_vector_new = np.matmul(R, np.array([1,0,0])) # cup_directional_vector w.r.t. global reference frame
#    c = np.dot(end_effector_orient_vector_last,end_effector_orientation_vector_new)/(np.linalg.norm(end_effector_orient_vector_last) * np.linalg.norm(end_effector_orientation_vector_new))
#    end_effector_cartesian_angle_difference = np.arccos(np.clip(c, -1, 1)) * 180 / ms.pi 
#    if returnCode != vrep.simx_return_ok:
#        print('Can not find Cup positions!')     
#    # ------------------------------------compute the particles in source container-----------------------------------
#    for i in range(len(particle_handle)):
#        returnCode, particle_position = vrep.simxGetObjectPosition(clientID, particle_handle[i], -1, vrep.simx_opmode_buffer)    
#        if returnCode!= vrep.simx_return_ok :
#            print('cannot get particle position.')
#        elif np.linalg.norm(np.array(particle_position) - np.array(cup_position)) >= distance_threshold:
#            step_spillage += 1     
#    return step_spillage, end_effector_position_difference, end_effector_cartesian_angle_difference, end_effector_orientation_vector_new, end_effector_position_new


def step_check_routine(clientID, cup_handle, end_effector_handle, particle_handle, distance_threshold):
    '''
    end_effector_orientation_vector : the normal vector of the end_effector in 3d
    '''
    step_spillage = 0
    returnCode, cup_position = vrep.simxGetObjectPosition(clientID, cup_handle, -1, vrep.simx_opmode_buffer)
    returnCode, end_effector_euler_angles = vrep.simxGetObjectOrientation(clientID, end_effector_handle, -1, vrep.simx_opmode_buffer)
    returnCode, end_effector_position = vrep.simxGetObjectPosition(clientID, end_effector_handle, -1, vrep.simx_opmode_buffer)    
    # Post-processing
    if returnCode != vrep.simx_return_ok:
        print('Can not find Cup positions!')     
    # ------------------------------------compute the particles in source container-----------------------------------
    for i in range(len(particle_handle)):
        returnCode, particle_position = vrep.simxGetObjectPosition(clientID, particle_handle[i], -1, vrep.simx_opmode_buffer)    
        if returnCode!= vrep.simx_return_ok :
            print('cannot get particle position.')
        elif np.linalg.norm(np.array(particle_position) - np.array(cup_position)) >= distance_threshold:
            step_spillage += 1     
    return step_spillage, np.array(end_effector_position), end_effector_euler_angles




def vrepEulerRotation(eulerAngles):
    a = eulerAngles[0]
    b = eulerAngles[1]
    c = eulerAngles[2]
    Rx = np.array([[1,0,0], [0,cos(a),-sin(a)], [0,sin(a),cos(a)]])
    Ry = np.array([[cos(b),0,sin(b)],[0,1,0], [-sin(b),0,cos(b)]])
    Rz = np.array([[cos(c),-sin(c),0], [sin(c),cos(c),0], [0,0,1]])
    R = np.matmul(Rx , Ry )
    R = np.matmul(R, Rz )
    return R