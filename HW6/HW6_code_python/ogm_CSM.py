# Occupancy Grid Mapping Counting Sensor Model Class
#
# Author: Chien Erh Lin, Fangtong Liu
# Date: 02/27/2021

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from utils import cart2pol, wrapToPI


# Occupancy Grid Mapping with Counting Sensor Model Class
class ogm_CSM:

    def __init__(self):
        # map dimensions
        self.range_x = [-15, 20]
        self.range_y = [-25, 10]

        # senesor parameters
        self.z_max = 30     # max range in meters
        self.n_beams = 133  # number of beams, we set it to 133 because not all measurements in the dataset contains 180 beams 

        # grid map parameters
        # self.grid_size = 0.135
        self.grid_size = 0.3
        self.w_obstacle = 2 * self.grid_size    # width of obstacle, 2 * grid_siz
        self.w_beam = 2 * np.pi / self.n_beams  # width of beam, 2 * pi/n_beams
        self.nn = 16                            # number of nearest neighbor search

        # map structure
        self.map = {}   # map
        self.pose = {}  # pose data
        self.scan = []  # laser scan data
        self.m_i = {}   # cell i

        # -----------------------------------------------
        # To Do: 
        # prior initialization
        # Initialize prior, prior_alpha
        # -----------------------------------------------
        self.prior = None            # prior for setting up mean and variance
        self.prior_alpha = 0.001      # a small, uninformative prior for setting up alpha

    def construct_map(self, pose, scan):
        # class constructor
        # construct map points, i.e., grid centroids
        x = np.arange(self.range_x[0], self.range_x[1]+self.grid_size, self.grid_size)
        y = np.arange(self.range_y[0], self.range_y[1]+self.grid_size, self.grid_size)
        X, Y = np.meshgrid(x, y)
        t = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
        # print("t = ", t)

        # a simple KDTree data structure for map coordinates
        self.map['occMap'] = KDTree(t)
        self.map['size'] = t.shape[0]

        # set robot pose and laser scan data
        self.pose['x'] = pose['x'][0][0]
        self.pose['y'] = pose['y'][0][0]
        self.pose['h'] = pose['h'][0][0] # Gussed: heading
        self.pose['mdl'] = KDTree(np.hstack((self.pose['x'], self.pose['y'])))
        self.scan = scan
        # print(self.scan[2][0].shape)

        # -----------------------------------------------
        # To Do: 
        # Initialization map parameters such as map['mean'], map['variance'], map['alpha'], map['beta']
        # -----------------------------------------------
        self.map['mean'] = (self.prior_alpha / 2 * self.prior_alpha) * np.ones((self.map['size'], 1))       # size should be (number of data) x (1)
        self.map['variance'] = (self.prior_alpha * self.prior_alpha / (self.prior_alpha + self.prior_alpha) ** 2 / self.prior_alpha + self.prior_alpha + 2) * np.ones((self.map['size'], 1))   # size should be (number of data) x (1)
        self.map['alpha'] = self.prior_alpha * np.ones((self.map['size'], 1))
        self.map['beta'] = self.prior_alpha * np.ones((self.map['size'], 1))


    def is_in_perceptual_field(self, m, p):
        # m = self.map['occMap'].data[i, :]
        # pose_k = np.array([self.pose['x'][k], self.pose['y'][k], self.pose['h'][k]])

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


    def counting_sensor_model(self, z, i):
        bearing_diff = []
        # find the nearest beam
        bearing_diff = np.abs(wrapToPI(z[:, 1] - self.m_i['phi']))
        k = np.nanargmin(bearing_diff)
        bearing_min = bearing_diff[k]

        # -----------------------------------------------
        # To Do: 
        # implement the counting sensor model, update obj.map.alpha and
        # obj.map.beta
        # Hint: the way to determine occupied or free is similar to
        # inverse sensor model
        # -----------------------------------------------
        if (self.m_i['range'] > min(self.z_max, z[k, 0] + self.w_obstacle / 2)) or (bearing_min > self.w_beam / 2):
            return (0, 0)
        elif (z[k, 0] < self.z_max) and (np.abs(self.m_i['range'] - z[k, 0]) < self.w_obstacle / 2):
            return (1, 0)
        elif (self.m_i['range'] < z[k, 0]) and (z[k, 0] < self.z_max):
            return (0, 1)


    def build_ogm(self):
        # build occupancy grid map using the binary Bayes filter.
        # We first loop over all map cells, then for each cell, we find
        # N nearest neighbor poses to build the map. Note that this is
        # more efficient than looping over all poses and all map cells
        # for each pose which should be the case in online (incremental)
        # data processing.
        for i in tqdm(range(self.map['size'])): # Iterate map cells
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
                        # print(z.shape)
                        # print("scan[k][0].shape = ", self.scan[k][0].shape)
                        # -----------------------------------------------
                        # To Do: 
                        # update the sensor model in cell i
                        # -----------------------------------------------
                        d_alpha, d_beta = self.counting_sensor_model(z, i)
                        self.map['alpha'][i] += d_alpha
                        self.map['beta'][i] += d_beta

            # -----------------------------------------------
            # To Do: 
            # update mean and variance for each cell i
            # -----------------------------------------------
            self.map['mean'][i] = (self.map['alpha'][i]) / (self.map['alpha'][i] + self.map['beta'][i])
            self.map['variance'][i] = (self.map['alpha'][i] * self.map['beta'][i]) / ((self.map['alpha'][i] + self.map['beta'][i]) ** 2 * (self.map['alpha'][i] + self.map['beta'][i] + 1))
