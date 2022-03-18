# Occupancy Grid Mapping Continuous Counting Sensor Model Class
#
# Author: Chien Erh Lin, Fangtong Liu
# Date: 02/27/2021

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from utils import cart2pol, wrapToPI


# Occupancy Grid Mapping with Continuous Counting Sensor Model Class
class ogm_continuous_CSM:

    def __init__(self):
        # map dimensions
        self.range_x = [-15, 20]
        self.range_y = [-25, 10]

        # senesor parameters
        self.z_max = 30     # max range in meters
        self.n_beams = 133  # number of beams, we set it to 133 because not all measurements in the dataset contains 180 beams 

        # grid map parameters
        # self.grid_size = 0.135  # adjust this for task 2.B
        self.grid_size = 1  # adjust this for task 2.B
        self.nn = 16            # number of nearest neighbor search

        # map structure
        self.map = {}   # map
        self.pose = {}  # pose data
        self.scan = []  # laser scan data
        self.m_i = {}   # cell i

        # continuous kernel parameter
        self.l = 0.2      # kernel parameter
        self.sigma = 0.1  # kernel parameter

        # self.num_samples = 20

        # -----------------------------------------------
        # To Do: 
        # prior initialization
        # Initialize prior, prior_alpha
        # -----------------------------------------------
        self.prior = None            # prior for setting up mean and variance
        self.prior_alpha = None      # a small, uninformative prior for setting up alpha

    def construct_map(self, pose, scan):
        # class constructor
        # construct map points, i.e., grid centroids
        x = np.arange(self.range_x[0], self.range_x[1]+self.grid_size, self.grid_size)
        y = np.arange(self.range_y[0], self.range_y[1]+self.grid_size, self.grid_size)
        X, Y = np.meshgrid(x, y)
        t = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

        # a simple KDTree data structure for map coordinates
        self.map['occMap'] = KDTree(t)
        self.map['size'] = t.shape[0]

        # set robot pose and laser scan data
        self.pose['x'] = pose['x'][0][0]
        self.pose['y'] = pose['y'][0][0]
        self.pose['h'] = pose['h'][0][0]
        self.pose['mdl'] = KDTree(np.hstack((self.pose['x'], self.pose['y'])))
        self.scan = scan

        # -----------------------------------------------
        # To Do: 
        # Initialization map parameters such as map['mean'], map['variance'], map['alpha'], map['beta']
        # -----------------------------------------------
        self.map['mean'] = 0.5 * np.ones((self.map['size'], 1))       # size should be (number of data) x (1)
        self.map['variance'] = 0.25 * np.ones((self.map['size'], 1))   # size should be (number of data) x (1)
        self.map['alpha'] = 0.001 * np.ones((self.map['size'], 1))
        self.map['beta'] = 0.001 * np.ones((self.map['size'], 1))


    def is_in_perceptual_field(self, m, p):
        # check if the map cell m is within the perception field of the
        # robot located at pose p
        inside = False
        d = m - p[0:2].reshape(-1)
        self.m_i['range'] = np.sqrt(np.sum(np.power(d, 2)))
        self.m_i['phi'] = wrapToPI(np.arctan2(d[1], d[0]) - p[2])
        # check if the range is within the feasible interval
        if (0 < self.m_i['range']) and (self.m_i['range'] < self.z_max):
            # here sensor covers -pi to pi
            if (-np.pi < self.m_i['phi']) and (self.m_i['phi'] < np.pi):
                inside = True
        return inside

    def kernel(self, d1):
        A = 1 / 3 * (2 + np.cos(2 * np.pi * d1 / self.l) * (1 - d1 / self.l))
        B = 1 / (2 * np.pi) * np.sin(2 * np.pi * d1 / self.l)
        return self.sigma * (A + B)
    
    
    def continuous_CSM(self, z, i, k):
        bearing_diff = []
        # find the nearest beam
        bearing_diff = np.abs(wrapToPI(z[:, 1] - self.m_i['phi']))
        idx = np.nanargmin(bearing_diff)
        global_x = self.pose['x'][k][0] + z[idx,0] * np.cos(z[idx,1] + self.pose['h'][k][0])
        global_y = self.pose['y'][k][0] + z[idx,0] * np.sin(z[idx,1] + self.pose['h'][k][0])

        # -----------------------------------------------
        # To Do: 
        # implement the continuous counting sensor model, update 
        # obj.map.alpha and obj.map.beta
        #
        # Hint: use distance and obj.l to determine occupied or free.
        # There might be multiple ways to update obj.map.beta. 
        # One way is to segment the measurement into several range 
        # values and update obj.map.beta if the distance is smaller 
        # than obj.l  
        # -----------------------------------------------
        d_alpha = 0
        d_beta = 0
        map_x = self.map['occMap'].data[i, 0]
        map_y = self.map['occMap'].data[i, 1]
        d1 = np.sqrt((global_x - map_x) ** 2 + (global_y - map_y) ** 2)
        if d1 < self.l:
            d_alpha += self.kernel(d1)
        
        # Sample points
        laser_length = z[idx, 0]
        # n_sample = np.floor(laser_length / self.step)
        self.step = self.l
        n_sample = int(laser_length // self.step)
        # print("n_sample = ", n_sample)
        # sample_x = self.pose['x'][k][0]
        # sample_y = self.pose['y'][k][0]
        # print("start = ", self.pose['x'][k][0])
        # print("end = ", global_x)
        for i in range(n_sample + 1):
            sample_x = self.pose['x'][k][0] + i * self.step * np.cos(z[idx,1] + self.pose['h'][k][0])
            # print("sample = ", sample_x)
            sample_y = self.pose['y'][k][0] + i * self.step * np.sin(z[idx,1] + self.pose['h'][k][0])
            d2 = np.sqrt((sample_x - map_x) ** 2 + (sample_y - map_y) ** 2)
            if d2 < self.l:
                d_beta += self.kernel(d2)

        return (d_alpha, d_beta)

    def build_ogm(self):
        # build occupancy grid map using the binary Bayes filter.
        # We first loop over all map cells, then for each cell, we find
        # N nearest neighbor poses to build the map. Note that this is
        # more efficient than looping over all poses and all map cells
        # for each pose which should be the case in online (incremental)
        # data processing.
        for i in tqdm(range(self.map['size'])):
            m = self.map['occMap'].data[i, :]
            _, idxs = self.pose['mdl'].query(m, self.nn)
            if len(idxs):
                for k in idxs:
                    # pose k
                    pose_k = np.array([self.pose['x'][k], self.pose['y'][k], self.pose['h'][k]])
                    if self.is_in_perceptual_field(m, pose_k):
                        # laser scan at kth state; convert from cartesian to
                        # polar coordinates
                        z = cart2pol(self.scan[k][0][0, :], self.scan[k][0][1, :])
                        # -----------------------------------------------
                        # To Do: 
                        # update the sensor model in cell i
                        # -----------------------------------------------
                        d_alpha, d_beta = self.continuous_CSM(z, i, k)
                        self.map['alpha'][i] += d_alpha
                        self.map['beta'][i] += d_beta

            # -----------------------------------------------
            # To Do: 
            # update mean and variance for each cell i
            # -----------------------------------------------
            self.map['mean'][i] = (self.map['alpha'][i]) / (self.map['alpha'][i] + self.map['beta'][i])
            self.map['variance'][i] = (self.map['alpha'][i] * self.map['beta'][i]) / ((self.map['alpha'][i] + self.map['beta'][i]) ** 2 * (self.map['alpha'][i] + self.map['beta'][i] + 1))
