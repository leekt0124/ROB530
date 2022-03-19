import numpy as np
import yaml
from scipy.io import loadmat
from data.generate_data import generateScript

class DataHandler:

    def __init__(self):
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        self.data_pth = param['data_path']

    def load_2d_data(self):
        out = {}

        deltaT = 0.1
        initialStateMean = [180,50,0]
        initialStateCov = np.eye(3)

        alphas = np.array([0.00025,0.00005,0.0025,0.0005,0.0025,0.0005])**2

        beta = 5/180*np.pi

        
        numSteps = 100
        data_generated = generateScript(initialStateMean, numSteps, alphas, beta, deltaT)

        # raw_data = loadmat(self.data_pth)
        
        # data = raw_data['data']
        data = data_generated
        self.data = data

        out['motionCommand'] = np.array(data[:,6:9]) # [Trans_vel,Angular_vel,gamma]' noisy control command
        out['observation_1'] = np.array(data[:,0:3]) # [bearing, range, landmark_id]' noisy observation
        out['observation_2'] = np.array(data[:,3:6]) # [bearing, range, landmark_id]' noisy observation
        out['observation'] = np.hstack((data[:,0:3], data[:,3:6]))
        out['Y'] = np.array(data[:,21:24]) # landmark 1 position relative to the robot
        out['Y2'] = np.array(data[:, 24:27]) # landmark 2 position relative to the robot

        out['actual_state'] = np.array(data[:,15:18])
        out['noise_free_state'] = np.array(data[:,18:21]) 

        out['noisefreeBearing_1'] = np.array(data[:, 9])
        out['noisefreeBearing_2'] = np.array(data[:, 12])

        return out
