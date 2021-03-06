
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:38:34 2019
To record:
    (1) step spillage
    (2) episodic spillage
    (3) Maximal rotated angle
    (4) Final resting angle

New:
Notice: Change the trajectory time also needs to change cs.py
@author: Honghu Xue
"""

import vrep
from dmp_discrete import DMPs_discrete
import sys
import numpy as np
import logging
import math as ms
import time
import cma
import pickle
import multiprocessing
from simulator import *
import matplotlib.pyplot as plt
import pandas as pd
from math import sin, cos
import os


def euler_angle_to_angular_velocity(euler_angles, conf):
    '''Input : a list of arrays [np.array[2,2,2]   , np.array[2,2,2]... ]
    Output : the same from as input
    all in radians
    '''
    #https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
    # P29
    angular_velocity = []   
    euler_angles.append(euler_angles[-1])
    euler_angles = np.array(euler_angles)
    change_in_euler_angles = np.diff(euler_angles, axis = 0) # one row less than euler angles
    for i in range(len(euler_angles)-1):
        E = np.array([[1,0,sin(euler_angles[i][1])], 
                      [0,cos(euler_angles[i][0]),-cos(euler_angles[i][1])*sin(euler_angles[i][0])],
                      [0,sin(euler_angles[i][0]),cos(euler_angles[i][1])*cos(euler_angles[i][0])]])        
        angular_velocity.append(np.matmul(E, change_in_euler_angles[i]) * 180/ (ms.pi * conf['dt']))
    return angular_velocity 


def reinitialization_process(conf):
    vrep.simxStopSimulation(conf['clientID'], vrep.simx_opmode_oneshot_wait)
    time.sleep(1.)
    vrep.simxSynchronous(conf['clientID'],True)
    return_code = vrep.simxStartSimulation(conf['clientID'], vrep.simx_opmode_oneshot)# However, v-rep will do strange initialization.
    while return_code != 0:
        return_code = vrep.simxStartSimulation(conf['clientID'], vrep.simx_opmode_oneshot) 
#    vrep.simxSetStringSignal(clientID,'jacoHand','true',vrep.simx_opmode_oneshot)    
    # starting streaming mode
    for i in range(conf['num_particles']):
        starting_streaming_buffer_mode(conf['clientID'], conf['particle_handles'][i], 'simxGetObjectPosition')
    starting_streaming_buffer_mode(conf['clientID'], conf['cup_handle'], 'simxGetObjectPosition')
    starting_streaming_buffer_mode(conf['clientID'], conf['cup_handle'], 'simxGetObjectOrientation')
    starting_streaming_buffer_mode(conf['clientID'], conf['end_effector_handle'], 'simxGetObjectOrientation')
    starting_streaming_buffer_mode(conf['clientID'], conf['end_effector_handle'], 'simxGetObjectPosition')
#    print('STREAMING_MODE FINISHED')
    for i in range(len(conf['joint_handles'])):
        returnCode = vrep.simxSetJointTargetPosition(conf['clientID'], conf['joint_handles'][i], conf['initial_joint_angles'][i], vrep.simx_opmode_streaming)
    if returnCode != vrep.simx_return_ok:
        print('Initialization of the joint angles fails.' + str(returnCode))   
    for i in range(round(0.03/conf['dt'])):
        return_code = vrep.simxSynchronousTrigger(conf['clientID'])
    # get cup position
    returnCode, cup_position = vrep.simxGetObjectPosition(conf['clientID'], conf['cup_handle'], -1, vrep.simx_opmode_buffer)
    if returnCode != 0:
        print('Can not find Cup positions!')          
    # Distribute different positions to particles according to the position of liquid
    for i in range(len(conf['particle_handles'])):
        returnCode = vrep.simxSetObjectPosition(conf['clientID'], conf['particle_handles'][i], -1, np.array([cup_position[0],cup_position[1],cup_position[2] + 0.022* i]), vrep.simx_opmode_oneshot)
    if returnCode != 0:
        print('Can not set particle position! Return code: %i' %returnCode)     
    for i in range(round(0.7/conf['dt'])):
        return_code = vrep.simxSynchronousTrigger(conf['clientID'])
        # Wait for the closing
    #time.sleep(5)


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))
 

def alpha_beta_rescaling(alpha, beta, conf):
    slope = 0.1
    if alpha >= (conf['alpha'][1] - conf['alpha'][0])/(2*slope):
        alpha_rescaled = conf['alpha'][1]
    elif alpha <= (-conf['alpha'][1] + conf['alpha'][0])/(2*slope):
        alpha_rescaled = conf['alpha'][0]
    else:
        alpha_rescaled =  slope * alpha + (conf['alpha'][1] + conf['alpha'][0])/2
    
    if conf['enable_beta'] == True:
        if beta >= (conf['beta'][1] - conf['beta'][0])/(2*slope):
            beta_rescaled = conf['beta'][1]
        elif beta <= (-conf['beta'][1] + conf['beta'][0])/(2*slope):
            beta_rescaled = conf['beta'][0]
        else:
            beta_rescaled =  slope * beta + (conf['beta'][1] + conf['beta'][0])/2      
    else:
        beta_rescaled = 0.25* alpha_rescaled
    return alpha_rescaled, beta_rescaled


def runtime_rescaling(time, conf):
    slope = 0.1
    if time >= (conf['run_time'][1] - conf['run_time'][0])/(2*slope):
        runtime_rescaled = conf['run_time'][1]
    elif time <= (-conf['run_time'][1] + conf['run_time'][0])/(2*slope):
        runtime_rescaled = conf['run_time'][0]
    else:
        runtime_rescaled =  slope * time + (conf['run_time'][1] + conf['run_time'][0])/2
    return runtime_rescaled


def DMP_trajectory_constraint_check(constr, joint_angle_traj, joint_vel_traj, joint_acc_traj):#, end_effector_velocity, end_effector_accel, end_effector_angular_velocity, end_effector_angular_accel):
    constraint_satisfied = True
    dist = 0
    enable_trajectory_clamping = False

    if (enable_trajectory_clamping == True):
        for i in range(len(joint_angle_traj)):
            key_angle = 'joint' + str(i + 1)
            key_velocity = 'joint' + str(i + 1)+ '_vel'
            key_accel = 'joint' + str(i + 1)+ '_accel'
            np.clip(joint_angle_traj[i], constr[key_angle][0], constr[key_angle][1], out = joint_angle_traj[i])
            np.clip(joint_vel_traj[i], constr[key_velocity][0], constr[key_velocity][1], out = joint_vel_traj[i])
            np.clip(joint_acc_traj[i], constr[key_accel][0], constr[key_accel][1], out = joint_acc_traj[i])
    else:          
        for i in range(len(joint_angle_traj)):
#            print(joint_angle_traj[i])
            max_angle = np.amax(joint_angle_traj[i])
            min_angle = np.amin(joint_angle_traj[i])
            max_vel = np.amax(joint_vel_traj[i])
            min_vel = np.amin(joint_vel_traj[i])
#            max_acc = np.amax(joint_acc_traj[i])
#            min_acc = np.amax(joint_acc_traj[i])
            key_angle = 'joint' + str(i + 1)
            key_velocity = 'joint' + str(i + 1)+ '_vel'
            key_accel = 'joint' + str(i + 1)+ '_accel' 
            cond_1 = min_angle >= constr[key_angle][0]
            cond_2 = max_angle <= constr[key_angle][1]
            cond_3 = min_vel >= constr[key_velocity][0]
            cond_4 = max_vel <= constr[key_velocity][1]
#            cond_5 = min_acc >= constr[key_accel][0]
#            cond_6 = max_acc <= constr[key_accel][1]
            
            if (cond_1) and (cond_2) and (cond_3) and (cond_4): #and (cond_5) and (cond_6): 
                constraint_satisfied = (True) and (constraint_satisfied)
            else:
                constraint_satisfied = False
                # distance to the mean value of feasible domain
                dist_1 = abs(min_angle - constr[key_angle][0])*(1 -float(cond_1)) 
                dist_2 = abs(max_angle - constr[key_angle][1])*(1- float(cond_2))
                dist_3 = abs(min_vel - constr[key_velocity][0])*(1- float(cond_3))
                dist_4 = abs(max_vel - constr[key_velocity][1])*(1 - float(cond_4))
#                dist_5 = abs(min_acc - constr[key_accel][0])*(1- float(cond_5))
#                dist_6 = abs(max_acc - constr[key_accel][1])*(1 - float(cond_6))
                dist = dist + dist_1 + dist_2 + dist_3 + dist_4# + dist_5 + dist_6
        
        if constraint_satisfied == False:
            dist /= float(len(joint_angle_traj))
        # Cartesian constraint
#        max_cartesian_vel = np.amax(end_effector_velocity)
#        max_cartesian_accel = np.amax(end_effector_accel)
#        max_cartesian_angular_vel = np.amax(end_effector_angular_velocity)
#        max_cartesian_angular_accel = np.amax(end_effector_angular_accel)
#        cond_7 = max_cartesian_vel <= constr['cartesian_vel_constr']
#        cond_8 = max_cartesian_accel <= constr['cartesian_accel_constr']  
#        cond_9 = max_cartesian_angular_vel <= constr['cartesian_angular_vel_constr']
#        cond_10 = True #max_cartesian_angular_accel <= constr['cartesian_angular_accel_constr']  #
#        constraint_satisfied = constraint_satisfied and cond_7 and cond_8 and cond_9 and cond_10        
        
#        dist_7 = abs(max_cartesian_vel - constr['cartesian_vel_constr'])*(1 -float(cond_7)) 
#        dist_8 = abs(max_cartesian_accel - constr['cartesian_accel_constr'])*(1 -float(cond_8)) 
#        dist_9 = abs(max_cartesian_angular_vel - constr['cartesian_angular_vel_constr'])*(1 -float(cond_9)) 
#        dist_10 = 0 #abs(max_cartesian_angular_accel - constr['cartesian_angular_accel_constr'])*(1 -float(cond_10))  
#        dist = dist + dist_7 + dist_8 + dist_9 + dist_10
#        print(dist_7 , dist_8, dist_9 , dist_10)
    return constraint_satisfied, dist * 1.0


def evaluation_and_stat(weights, conf):   
    '''
    Assume time is fixed : t = 5s
    sampling_rate = how many samples per second is sent to simulator
    redundant_simulation_step = how many additional simulation steps are necessary for reaching stable system state.
    ''' 
    spillage_distance_threshold = 0.11
    redundant_simulation_step = round(1.25/conf['dt']) # 1.25s
    linear_mode = False
    worst_cost_function_value = 180 * 1.2 + 2 * float(len(conf['particle_handles'])) * (80/20)# as collision is one time such cost
    collision = False
    add_noise = True
    if linear_mode == True:
        total_time = 1.
        sampling_rate = 1./conf['dt']
        def sample_points_linear(sampling_rate, current_time_step, weights):
            """
            weights = [a1,...a7]
            """  
            current_joint_angles = np.zeros(len(weights))              
            for i in range(len(weights)):
                if i == 0:
                    current_joint_angles[i] = weights[i] * current_time_step / sampling_rate + ms.pi/2
                elif i == 1:
                    current_joint_angles[i] = weights[i] * current_time_step / sampling_rate - ms.pi/2
                else:
                    current_joint_angles[i] = weights[i] * current_time_step / sampling_rate
            return current_joint_angles        
        for t in range(round(total_time * sampling_rate)):            
            theta = sample_points_linear(sampling_rate, t, weights)
            for i in range(len(conf['joint_handles'])):
                returnCode = vrep.simxSetJointTargetPosition(conf['clientID'], conf['joint_handles'][i], theta[i], vrep.simx_opmode_streaming) #Original: vrep.simx_opmode_blocking
            vrep.simxSynchronousTrigger(conf['clientID'])
            if returnCode != 0:
                print('Can not set joint target positions!')
            constraint_satisfied = True 
    else:# -----------------------------------------------------------------DMP mode-------------------------------------------------------------
        joint_angle_traj, joint_vel_traj, joint_acc_traj, angle_to_vertical_axis, step_spillage = [] , [] , [] , [] , [] 
#        end_effector_position, end_effector_euler_angles = [] , []
        n_dmps = 1     
        rollout_timesteps = conf['dmp_trajectory_timesteps']# Originally, 5 seconds, now even different among different joint angles.

        for i in range(len(conf['joint_handles'])):
            ##Allowing alpha and beta learning 
            if conf['enable_beta'] == True:  
                alpha_rescaled, beta_rescaled = alpha_beta_rescaling(weights[i][-2], weights[i][-1], conf)
            else:
                alpha_rescaled, beta_rescaled = alpha_beta_rescaling(weights[i][-1], weights[i][-2], conf)

            # Goal position allowed
            dmp = DMPs_discrete(dt=conf['dt_dmp'], n_dmps=n_dmps, n_bfs=conf['n_bfs'], w=np.expand_dims(weights[i][:-2], axis=0), goal=weights[i][-2]+conf['initial_joint_angles'][i]*180 /ms.pi, ay=alpha_rescaled*np.ones(n_dmps), by=beta_rescaled*np.ones(n_dmps), y0=conf['initial_joint_angles'][i]*180 /ms.pi)   # weight as parameters, goal as known            
            # Goal position fixed
#            dmp = DMPs_discrete(dt=conf['dt_dmp'], n_dmps=n_dmps, n_bfs=conf['n_bfs'], w=np.exp(np.expand_dims(weights[i][:-1], axis=0)), goal=conf['initial_joint_angles'][i]*180 /ms.pi, ay=alpha_rescaled*np.ones(n_dmps), by=beta_rescaled*np.ones(n_dmps), y0=conf['initial_joint_angles'][i]*180 /ms.pi)   # weight as parameters, goal as known            

#            # rescale the runtime
#            if conf['enable_beta'] == False:
#                dmp.cs.run_time = runtime_rescaling(weights[i][-1], conf)
               
            y_track, dy_track, ddy_track = dmp.rollout(timesteps = rollout_timesteps, tau = 1.0) # Makes sure that the trajectory lengths are the same.     #dmp.rollout(timesteps = total_time_for_dyms) 
            # Post-process the y_track and dy_track
            y_track = y_track[0::round(conf['dt']/conf['dt_dmp'])]
            dy_track = np.mean(dy_track.reshape(-1, round(conf['dt']/conf['dt_dmp'])), axis=1)
            ddy_track = np.mean(ddy_track.reshape(-1, round(conf['dt']/conf['dt_dmp'])), axis=1)
            if add_noise == True:
                y_track += 0.2 * np.random.randn(*y_track.shape)
#                dy_track += 0.2 * np.random.randn(*dy_track.shape)
            joint_angle_traj.append(y_track)
            joint_vel_traj.append(dy_track)
            joint_acc_traj.append(ddy_track)     
            
        # ----------------------------------------Check boundaries is fulfilled or not---------------------------------------------            
        constraint_satisfied, constraint_punishment = DMP_trajectory_constraint_check(conf, joint_angle_traj, joint_vel_traj, joint_acc_traj)
        
        # -----------------------------------------------------------Start simulation------------------------------------------------------                   
        if constraint_satisfied == True: 
            for t in range(len(y_track)):                
                vrep.simxPauseCommunication(conf['clientID'],True) 
                for i in range(len(conf['joint_handles'])):
                    returnCode = vrep.simxSetJointTargetPosition(conf['clientID'], conf['joint_handles'][i], joint_angle_traj[i][t]/180.*ms.pi, vrep.simx_opmode_streaming) #Original: vrep.simx_opmode_blocking
                    if returnCode != 0 :
                        print('Can not set joint target positions!' + str(returnCode))               
                vrep.simxPauseCommunication(conf['clientID'],False)
                vrep.simxSynchronousTrigger(conf['clientID'])                 
                if t == 0:
                    vrep.simxGetObjectOrientation(conf['clientID'], conf['cup_handle'], -1,  vrep.simx_opmode_streaming)
                    startTime=time.time()
                    while time.time()-startTime < 5:
                        returnCode, cup_eulerAngles = vrep.simxGetObjectOrientation(conf['clientID'], conf['cup_handle'], -1, vrep.simx_opmode_buffer)
                        if returnCode==vrep.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
                            break
                            time.sleep(0.01)
                else:
                    returnCode, cup_eulerAngles = vrep.simxGetObjectOrientation(conf['clientID'], conf['cup_handle'], -1, vrep.simx_opmode_buffer)
                    if returnCode != 0:
                        print('Can not get cup orientation!' + str(returnCode))
                    R = vrepEulerRotation(cup_eulerAngles)
                    cup_directional_vector = np.matmul(R, np.array([0,0,1])) # cup_directional_vector w.r.t. global reference frame
        #            Compute angle between 2 vectors
        #            https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
                    c = np.dot(cup_directional_vector,np.array([0, 0, -1]))/np.linalg.norm(cup_directional_vector)
                    angle_to_vertical_axis.append(np.arccos(np.clip(c, -1, 1)) * 180 / ms.pi )
                
                #-------------------------------------------------------Collision Check -------------------------------------
                for i in range(len(conf['collision_handle'])):
                    if t == 0:
                        vrep.simxReadCollision(conf['clientID'], conf['collision_handle'][i], vrep.simx_opmode_streaming)
                        startTime=time.time()
                        while time.time()-startTime < 5:
                            returnCode,collision_temp=vrep.simxReadCollision(conf['clientID'], conf['collision_handle'][i], vrep.simx_opmode_buffer)
                            if returnCode==vrep.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
                                break
                            time.sleep(0.01)
                    else:
                        returnCode,collision_temp=vrep.simxReadCollision(conf['clientID'], conf['collision_handle'][i], vrep.simx_opmode_buffer)                          
    #                    returnCode, collision_temp = vrep.simxReadCollision(conf['clientID'], conf['collision_handle'][i], vrep.simx_opmode_buffer)
                    collision = collision or collision_temp
                if returnCode != vrep.simx_return_ok:
                    print('Can not read collsion state!' + str(returnCode))
                step_spillage_cumulative, end_effector_pose, end_effector_euler_angle = step_check_routine(conf['clientID'], conf['cup_handle'], conf['end_effector_handle'], conf['particle_handles'], conf['distance_threshold'])
                step_spillage.append(step_spillage_cumulative)
#                if collision == True:
#                    break
            
    if (constraint_satisfied == True): #and (min(angle_to_vertical_axis) < 90):        
        returnCode,Target_Position = vrep.simxGetObjectPosition(conf['clientID'],conf['cup_handle'],-1,vrep.simx_opmode_buffer)
        if returnCode != 0:
            print('Can not get Target position!')
        if collision == False:
            for j in range(redundant_simulation_step):
                for i in range(len(conf['joint_handles'])):
                    returnCode = vrep.simxSetJointTargetPosition(conf['clientID'], conf['joint_handles'][i], joint_angle_traj[i][-1]/180.*ms.pi, vrep.simx_opmode_streaming) #Original: vrep.simx_opmode_blocking
                vrep.simxSynchronousTrigger(conf['clientID'])     
        total_spilled_amount = episodic_spillage_detection(conf['clientID'], conf['cup_handle'], conf['particle_handles'], spillage_distance_threshold)       
#        e = np.linalg.norm((conf['p_d'] - Target_Position), ord=2) # no longer needed for turnover
        e = total_spilled_amount * (80/20) + min(angle_to_vertical_axis) + 0.2 * (180 - angle_to_vertical_axis[-1])
        if collision == True:
            print('colliding')
            e = float(collision) * (80 * 1.2 + float(len(conf['particle_handles'])) * 80 /20) + total_spilled_amount * (80/20)
    else:
        e = worst_cost_function_value + constraint_punishment        
    
    record_step_spillage = constraint_satisfied and (not collision) # If one of the criterion is true, then don't record step spillage
    record_episodic_spillage = constraint_satisfied
    if record_step_spillage == True:
        # Process the cumulative information to increamental information
        step_spillage_tmp = step_spillage.copy()
        step_spillage.append(step_spillage_cumulative) # 
        step_spillage_tmp.insert(0, step_spillage[0])
        step_spillage_tmp = np.array(step_spillage_tmp)
        step_spillage = np.array(step_spillage)
        step_spillage = step_spillage - step_spillage_tmp        
    else:
        step_spillage = 25 * np.ones(rollout_timesteps)
    if record_episodic_spillage == True:
        episodic_spillage = total_spilled_amount
        max_cup_rotation_angle = 180 - min(angle_to_vertical_axis)
        final_cup_resting_angle = 180 - angle_to_vertical_axis[-1]  
    else:
        episodic_spillage = 25 
        max_cup_rotation_angle = 200
        final_cup_resting_angle = -25
                   
    return e, record_episodic_spillage, episodic_spillage, record_step_spillage, step_spillage, max_cup_rotation_angle, final_cup_resting_angle


def worker(port_num, input_queue, output_queue, conf, logger):   
    name = multiprocessing.current_process().name  
    clientID = start_simulator(port_num) 
    if clientID!=-1:             
        joint_handles = np.zeros((conf['num_joints'],1),dtype=np.float64)               
        particle_handles = np.zeros((conf['num_particles'],1),dtype=np.float64)    
        # get object handle for joints
        for i in range(1,conf['num_joints']+1):
            stringJoint = 'LBR_iiwa_14_R820_joint' +str(round(i)) 
            print(stringJoint)
            returnCode,joint_handles[i-1] = vrep.simxGetObjectHandle(clientID,stringJoint, vrep.simx_opmode_oneshot_wait)  
            if returnCode < 0:
                print('Can not find all joint handles!'  + str(returnCode))
                logger.error('Can not find all joint handles!'  + str(returnCode))
        print(joint_handles)
        #get end-effector handle
        returnCode, end_effector_handle = vrep.simxGetObjectHandle(clientID, 'pandaHand_attachment', vrep.simx_opmode_blocking)        
        # get object handle for particles
        for i in range(conf['num_particles']):
            # From Sphere0  to Sphere19
            Particle = 'Sphere' +str(round(i))
            returnCode,particle_handles[i] = vrep.simxGetObjectHandle(clientID,Particle, vrep.simx_opmode_blocking)
            if returnCode < 0:
                logger.warning('Can not find particle handles!  + str(returnCode)')
        # get object handle for cup
        returnCode,cup_handle = vrep.simxGetObjectHandle(clientID, 'Cup0', vrep.simx_opmode_blocking) 
        if returnCode != 0:           
            print('Can not find cup handles!  ' + str(returnCode))            
        collision_handle = []             
        returnCode, collision_handle1 = vrep.simxGetCollisionHandle(clientID, "Collision", vrep.simx_opmode_blocking)
        returnCode, collision_handle2 = vrep.simxGetCollisionHandle(clientID, "Collision0", vrep.simx_opmode_blocking)
        collision_handle.append(collision_handle1)
        collision_handle.append(collision_handle2)
        if returnCode != 0:           
            logger.error('Can not find collision handles!  ' + str(returnCode))   
        conf.update({'clientID':clientID, 'joint_handles':joint_handles, 'collision_handle':collision_handle, 'end_effector_handle': end_effector_handle, 'cup_handle':cup_handle, 'particle_handles':particle_handles})

        # ----------------------------------------------Initialization is done -----------------------------------------------
        while True:      
            reinitialization_process(conf)  
            query_point = input_queue.get()   # This will automatically block until the worker gets the content
            query_point = query_point.reshape((conf['num_joints'],-1))             
    
            # -----------------------------------------------start evaluation-----------------------------------------------------            
            cost, record_episodic_spillage, episodic_spillage, record_step_spillage, step_spillage, max_cup_rotation_angle, final_cup_resting_angle = evaluation_and_stat(query_point, conf)          
            # transmit the answer to main process 
            print(name + '\'s cost: %f' %cost) 
            print('Max_cup_roataion_angle : {}'.format(max_cup_rotation_angle))
            print('Final angle : {}'.format(final_cup_resting_angle))
            print('episodic_spillage : {}'.format(episodic_spillage))
            query_point = query_point.reshape((1, -1)).squeeze() 
            output_queue.put([query_point[0], cost, record_episodic_spillage, episodic_spillage, record_step_spillage, step_spillage, max_cup_rotation_angle, final_cup_resting_angle ])    
    else:
        print('Connection failed')
        sys.exit('could not connect') # Let python script exit      
        

def windows_open_multiple_Vrep_instances(conf, num_process, starting_port_number): 
    for i in range(num_process):
        # Modify txt
        with open(conf['remoteapiconnections_path'], 'w') as writer:
            writer.write('// This file defines all the continuous remote API server services (started at remote API plugin initialization, i.e. V-REP start-up) \n')
            writer.write('// Each remote API server service requires following 3 entries:\n')
            writer.write('// portIndex@_port = xxxx               // where xxxx is the desired port number (below 19997 are preferred for server services starting at V-REP start-up)\n')
            writer.write('// portIndex@_debug = xxxx              // where xxxx is true or false\n')
            writer.write('// portIndex@_syncSimTrigger = xxxx     // where xxxx is true or false. When true, then the service will be pre-enabled for synchronous operation.\n')
            writer.write('// In above strings, @ can be any number starting with 1. If more than one server service is required, then numbers need to be consecutive and starting with 1\n')
            writer.write('portIndex1_port             = ' + str(starting_port_number + i) + '\n')
            writer.write('portIndex1_debug            = false'+ '\n')
            writer.write('portIndex1_syncSimTrigger   = true'+ '\n')
        time.sleep(1.5)
        os.system('start ' + conf['simulator_path']) 
        time.sleep(8)


if __name__ == '__main__':     
    logger = logging.getLogger(__name__)  
    # set log level
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('logfile.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    validation = True
    resume_training = False
    num_process = 1
    save_step_spillage_every_n_iterations = 10
    starting_port_number = 19991
    CMA_variance = 10
    max_iter = 100 # not used now
    jobs = []    
    iteration_count = 0
    input_queue, output_queue = multiprocessing.Queue() , multiprocessing.Queue()
# --------------------------------------------------------------For dynamic movement primitives --------------------------------------------------   
    conf = {'simulator_path': 'C:\\PhD\\V-rep\\Turnover-Kuka-R820.ttt',
            'remoteapiconnections_path': 'C:\\Program Files\\V-REP3\\V-REP_PRO_EDU\\remoteApiConnections.txt', 
            'n_bfs': 10, 
            'num_joints':7, 
            'num_particles':20,             
            'dt':.01,  # The simulation step time in V-rep
            'dt_dmp':.01,  # The minimal resolution of DMPs------
            #'p_d':np.array([-0.0011, -0.6912, 1.5377]), no longer needed for turnover motion
            'initial_joint_angles': np.array([0., ms.pi/6, 0., -ms.pi/6, 0., ms.pi/6, 0]),
            'enable_alpha_beta': True,
            'distance_threshold': 0.11
            }  #np.array([0.01, 1.0914, 1.1377]) --> rightmost

    population_size = 120  #int( 4 +  np.floor(3 * np.log(num_joints * (n_bfs + 1))))  #num_joints * (n_bfs + 1) * 5
    print('population size : %i'  %population_size)    
    k = 180/ms.pi
    margin_deg = 5
    margin_vel, margin_acc = 0.92, 0.92
    conf.update({'joint1':[-170.+margin_deg, 170.-margin_deg], 'joint2':[-120.+margin_deg,120.-margin_deg], 'joint3':[-170.+margin_deg,170.-margin_deg], 'joint4':[-120.+margin_deg,120.-margin_deg], 'joint5':[-170.+margin_deg,170.-margin_deg], 'joint6':[-120.+margin_deg,120.-margin_deg], 'joint7':[-175.+margin_deg,175.-margin_deg],\
                 'joint1_vel':[-85.*margin_vel,85.*margin_vel], 'joint2_vel':[-85.*margin_vel,85.*margin_vel], 'joint3_vel':[-100.*margin_vel,100.*margin_vel], 'joint4_vel':[-75.*margin_vel,75.*margin_vel], 'joint5_vel':[-130.*margin_vel,130*margin_vel], 'joint6_vel':[-135.*margin_vel,135.*margin_vel], 'joint7_vel':[-135.*margin_vel,135.*margin_vel],\
#                 'joint1_accel':[-15*k*margin_acc, 15*k*margin_acc], 'joint2_accel':[-7.5*k*margin_acc, 7.5*k*margin_acc], 'joint3_accel':[-10.*k*margin_acc,10.*k*margin_acc], 'joint4_accel':[-12.5*k*margin_acc,12.5*k*margin_acc], 'joint5_accel':[-15*k*margin_acc,15*k*margin_acc], 'joint6_accel':[-20*k*margin_acc,20*k*margin_acc], 'joint7_accel':[-20*k*margin_acc,20*k*margin_acc],\
#                 'cartesian_angular_vel_constr': 2.5*k*margin_vel, 'cartesian_angular_accel_constr': 25*k*margin_acc, 'cartesian_vel_constr': 1.7*margin_vel, 'cartesian_accel_constr': 13*margin_acc,
                 'alpha':[1., 20.], 'beta':[1.,20.], 'enable_beta':False, 'run_time':[3.,5.]})
                # Note: alpha must be exponentiated.
    conf.update({'dmp_trajectory_timesteps':round(conf['run_time'][1]/conf['dt_dmp'])})
#-------------------------------------------------------------------------------------------------------------------------------------------------    
#    windows_open_multiple_Vrep_instances(conf, num_process, starting_port_number)
    if validation == True:
        xbest = np.array([   7.55228766,  -39.60924795,  -61.24545065,   40.50240254,
                            -46.96553908,   27.78052169,  -63.35194282,   17.7107209 ,
                             -6.86632143,  -91.254643  ,   27.13494881,  -56.43854817,
                            -11.07434888,    3.08218785,  -79.63259905,   67.0686615 ,
                             79.28459858,  -29.25695536,   -5.39950335,   36.1904218 ,
                            -56.57670308,   -5.45786663,   47.47943295,    1.01477826,
                            -19.54893822,   21.7075832 , -117.70195494,  121.90710026,
                             53.66145606,   77.95421516,  -45.25239064,  -41.89259458,
                            -62.78609946,   15.16508726,   33.2667007 ,    0.60402863,
                            -30.51343834,  -61.52874626,   24.75093361,  -39.30189632,
                            -15.25977756,  -28.88382383,   40.20122841,    1.63164943,
                            -16.88154138,  -38.81255306,   22.19400623,  -12.15522841,
                            -10.05602996,  -32.04014699, -171.77945223,   25.09747152,
                            -64.36007445,    8.59326816,   84.27459019,  -12.19204127,
                              5.76908386,  -35.05917698,   42.06457959,   -6.90086421,
                            -23.53883705,   16.44796389,  126.58104886,  -80.25364397,
                             37.83469718,   13.5464455 ,  -16.16823133,  -57.95129774,
                            -48.15781238,   32.51296595,   -6.90326565,  -28.62140791,
                             -9.17655398,    2.4286019 ,   72.68827312,  -82.43795429,
                             55.20012098,   15.67926873,  -82.6540482 ,  -85.60585423,
                              5.77768111,  -22.53311108,  -75.63233016,    4.41389782])
        
        for i in range(num_process):
            port_num = starting_port_number + i
            sim_process = multiprocessing.Process(target = worker, args = (port_num, input_queue, output_queue, conf, logger))
            jobs.append(sim_process)
            sim_process.start()
        input_queue.put(xbest)
        sys.exit('Program ends')
    
    for i in range(num_process):
        port_num = starting_port_number + i
        sim_process = multiprocessing.Process(target = worker, args = (port_num, input_queue, output_queue, conf, logger))
        jobs.append(sim_process)
        sim_process.start()

    # target position + alpha as additional parameters to be optimized. -----------------------TO DO: set bounds---------------------------

    params = np.zeros((1, conf['num_joints'] * (conf['n_bfs'] + 2)),dtype=np.float64).squeeze() # initial value for PID gains and passing it to optimizer CMA_ES
    # [w1,...w6, alpha]
    
    # Set the initial value for alpha, beta , not needed anymore
#    for i in range(conf['num_joints']):    
#        params[(i + 1) * (conf['n_bfs'] + 1) - 1] = 0

    logger.info('Initial parameter')
    logger.info(params)
    if resume_training == False:
        opts = cma.CMAOptions()
        for key, val in opts.items():
            print(key)        
        # save the result
        df = pd.DataFrame(columns = ['fitness_value', 'best_parameter', 'sigma', 'episodic_spillage', 'max_rotated_angle', 'final_angle']) 
        df2 = pd.DataFrame(columns = ['step_spillage']) 
        es = cma.CMAEvolutionStrategy(params.tolist(), CMA_variance, {'ftarget':1e-3, 'seed':92 ,'popsize': population_size})#, {'maxiter':max_iter, 'bounds': [-0.5, 0.5],  'seed':59})     
    else:
        es = pickle.load(open('C:\\PhD\\V-rep\\cma_DMP.pkl', 'rb'))
        df = pickle.load(open('C:\\PhD\\V-rep\\statistics.pkl', 'rb'))
        df2 = pickle.load(open('C:\\PhD\\V-rep\\statistics_step_spillage.pkl', 'rb'))
    while not es.stop():
        solutions = es.ask() 
        print('sigma: ',es.sigma)
        iteration_count += 1
        #--------------------------------------------------------------------------------------------------------------
        answers, outputs, out, episodic_spillage, max_rotated_angle, final_angle, step_spillage= [], [], [], [], [], [] , []
        for i in range(len(solutions)):
            port_num = 19991 + i
            input_queue.put(solutions[i])   
        while True:
            output = output_queue.get() # automatically block
            out.append(output[0:2])
            outputs.append(output)
            time.sleep(0.005)
            if len(outputs) == population_size:
                print('ALL answers at hand!')
#                print(outputs)
                break
        # ---sort the answers----
        for i in range(population_size):
            loc = index_2d(out, solutions[i][0])
            answers.append(outputs[loc[0]][1])
            
            if outputs[loc[0]][2] == True: # Record_episodic_spillage is true
                episodic_spillage.append(outputs[loc[0]][3])
            if outputs[loc[0]][4] == True:
                step_spillage.append(outputs[loc[0]][5])
                max_rotated_angle.append(outputs[loc[0]][-2])
                final_angle.append(outputs[loc[0]][-1]) 
        #--------------------------------------------------------------------------------------------------------------
        
        es.tell(solutions, answers)    
#        es.disp()   
#        es.sp.disp()
#        es.init('c1', val=0.00666666, warn=True) 
        answers.sort() 
        print('Answers : ' ,answers)       
        logger.info(answers)
        pickle.dump(es, open('cma_DMP.pkl', 'wb'))
        
        
        # save statistics
        df.loc[len(df)] = [answers, es.result[0].squeeze(), es.sigma,episodic_spillage , max_rotated_angle, final_angle]
        df.to_pickle("C:\\PhD\\V-rep\\statistics.pkl")
        if (iteration_count% save_step_spillage_every_n_iterations  == 0):
            df2.loc[len(df2)] = [step_spillage]
            df2.to_pickle("C:\\PhD\\V-rep\\statistics_step_spillage.pkl")
        
        
        print(es.result[0].reshape(conf['num_joints'],-1))
        logger.info(es.result)
        
        # If encountering a plateau
#        if abs(answers[0] - answers[round(len(answers)/3)]) <= 1e-2 and answers[0] > 1e-3:
#            es.sigma *= 1.5
 
    for i in range(population_size):
        jobs[i].terminate()
        logger.info("Subprocess %i has ended." %i)