
import sys
sys.path.append('.')
from os import stat
import numpy as np
from numpy.random import default_rng
rng = default_rng()
from utils.utils import wrap2Pi
from world.world2d import world2d

def prediction(state, motion):
    x = state[0]
    y = state[1]
    theta = state[2]
    v = motion[0]
    w = motion[1]
    gamma = motion[2]
    if w==0:
        x = x + np.cos(theta)*v
        y = y + np.sin(theta)*v
        theta = theta + w + gamma
    else:
        x = x+(-v/w*np.sin(theta)+v/w*np.sin(theta+w))
        y = y+(v/w*np.cos(theta)-v/w*np.cos(theta+w))
        theta = theta+w+gamma
    state = np.array([x,y,theta])
    state[2] = wrap2Pi(state[2])
    return state

def generate_motion(t, deltaT):
    n = t/deltaT
    index = n % (np.floor(1/deltaT)*5)

    if index == 0:
        m = [0,deltaT*100,0]
    elif index == 1*np.floor(1/deltaT):
        m = [0,deltaT*100,0]
    elif index == 2*np.floor(1/deltaT):
        m = [45/180*np.pi,deltaT*100,45/180*np.pi]
    elif index == 3*np.floor(1/deltaT):
        m = [0,deltaT*100,0]
    elif index == 4*np.floor(1/deltaT):
        m = [45/180*np.pi,0,45/180*np.pi]
    else:
        m = [0,deltaT*100,0]
    return m

def sample_odometry(motion, state, alphas):
    trans_vel = motion[0]
    angular_vel = motion[1]

    noisy_motion = np.zeros(3)
    noisy_motion[0] = rng.normal(motion[0], np.sqrt(alphas[0]*trans_vel**2+alphas[1]*angular_vel**2))
    noisy_motion[1] = rng.normal(motion[1], np.sqrt(alphas[2]*trans_vel**2+alphas[3]*angular_vel**2))
    noisy_motion[2] = rng.normal(motion[2], np.sqrt(alphas[4]*trans_vel**2+alphas[5]*angular_vel**2))
    
    state = prediction(state, noisy_motion)

    return state, noisy_motion

def pose_mat(X):
    x = X[0]
    y = X[1]
    h = X[2]
    H = np.array([[np.cos(h),-np.sin(h),x],\
                    [np.sin(h),np.cos(h),y],\
                    [0,0,1]])
    return H

def observation(state, id, FIELD_INFO):
    dx = FIELD_INFO.marker_pos[id,0] - state[0]
    dy = FIELD_INFO.marker_pos[id,1] - state[1]
    
    obs = np.array([wrap2Pi(np.arctan2(dy,dx)-state[2]), np.sqrt(dx**2+dy**2), id])

    return obs

def generateScript(initialStateMean, numSteps, alphas, beta, deltaT):
    observationDim = 3 # observation size (range, bearing, marker ID)

    realRobot = initialStateMean
    noisefreeRobot = initialStateMean

    data = np.zeros((numSteps,27))
    FIELD_INFO = world2d()

    
    for n in range(numSteps):
        t = n*deltaT
        noisefreeMotion = generate_motion(t, deltaT)

        dx = np.cos(noisefreeRobot[2] + noisefreeMotion[0]) * noisefreeMotion[1]
        dy = np.sin(noisefreeRobot[2] + noisefreeMotion[0]) * noisefreeMotion[1]
        linear_vel = np.sqrt(dx**2 + dy**2)
        angular_vel = noisefreeMotion[0] + noisefreeMotion[2]
        noisefreeMotion = [linear_vel, angular_vel, 0]
        noisefreeRobot,_ = sample_odometry(noisefreeMotion, noisefreeRobot, np.zeros(6))

        realRobot, noisymotion = sample_odometry(noisefreeMotion, realRobot, alphas)

        
        markerID = int(np.floor(n/2) % FIELD_INFO.num_landmarks_)
        markerID2 = int(np.floor(n/2) % FIELD_INFO.num_landmarks_) + 1
        if markerID2 == 6:
            markerID2 = 0
        landmark_x = FIELD_INFO.marker_pos[markerID,0]
        landmark_y = FIELD_INFO.marker_pos[markerID,1]
        landmark_x2 = FIELD_INFO.marker_pos[markerID2,0]
        landmark_y2 = FIELD_INFO.marker_pos[markerID2,1]

        b = np.array([landmark_x, landmark_y, 1])
        b2 = np.array([landmark_x2, landmark_y2, 1])
        
        groud_truth = pose_mat(realRobot)

        N = 100 * np.eye(2)
        LN = np.linalg.cholesky(N)

        temp = np.zeros((3,1))
        temp[:2,0] = (LN @ rng.standard_normal((2,1))).reshape(2)
        temp2 = np.zeros((3,1))
        temp2[:2,0] = (LN @ rng.standard_normal((2,1))).reshape(2)
        Y = np.linalg.inv(groud_truth) @ b.reshape(-1,1) + temp.reshape(-1,1)
        Y2 = np.linalg.inv(groud_truth) @ b2.reshape(-1,1) + temp2.reshape(-1,1)

        noisefree_observation = observation(realRobot, markerID, FIELD_INFO)
        noisefree_observation2 = observation(realRobot, markerID2, FIELD_INFO)

        Q = np.array([[beta**2,0,0],[0,100,0],[0,0,0]])

        observation_noise = rng.multivariate_normal(np.zeros(observationDim),Q)
        observation_noise2 = rng.multivariate_normal(np.zeros(observationDim),Q)
        real_observation = noisefree_observation + observation_noise
        real_observation2 = noisefree_observation2 + observation_noise2


        # TODO: to match the convention of matlab id
        # @Justin remove this and remove the +1 operation when assigning marker id
        noisefree_observation[2] += 1
        noisefree_observation2[2] += 1
        real_observation[2] += 1
        real_observation2[2] += 1

        data[n,:] = np.hstack((real_observation,real_observation2,noisymotion,noisefree_observation,\
            noisefree_observation2,realRobot,noisefreeRobot,Y.reshape(3),Y2.reshape(3)))




    return data

# from scipy.io import loadmat
# data = loadmat('./data/data2.mat')

# deltaT = 0.1
# initialStateMean = [180,50,0]
# initialStateCov = np.eye(3)

# alphas = np.array([0.00025,0.00005,0.0025,0.0005,0.0025,0.0005])**2

# beta = 5/180*np.pi
# numSteps = 100
# data_generated = generateScript(initialStateMean, numSteps, alphas, beta, deltaT)
# print(data_generated - data['data'])
