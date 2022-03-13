from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)



    def prediction(self, u):
        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        # Compute L'
        self.sigma_point(X.reshape(-1, 1), P, self.kappa_g)
        X_pred = np.zeros_like(X, dtype=float)
        P_pred = np.zeros_like(P, dtype=float)

        # print("self.X = ", self.X)
        # print("self.w = ", self.w)
        # np.apply_along_axis(self.gfun(u), 0, self.X)
        for i in range(2 * self.n + 1):
            # print(i)
            # self.X[:, i] = self.gfun(self.X[:, i], u)
            X_pred += self.w[i] * self.gfun(self.X[:, i], u)
            # print("new = ", self.w[i] * self.gfun(self.X[:, i], u))
            # print("X_pred = ", X_pred)
        
        # X_pred = np.mean(self.X * self.w, axis=1)
        # X_pred /= (self.n * 2 + 1)
        # print("X_pred.shape = ", X_pred)

        for i in range(2 * self.n + 1):
            # print(i)
            # print("self.X[:, i] = ", (self.X[:, i] - X_pred).reshape(-1, 1))
            P_pred += (self.gfun(self.X[:, i], u) - X_pred).reshape(-1, 1) @ (self.gfun(self.X[:, i], u) - X_pred).reshape(-1, 1).T  * self.w[i]
        P_pred += self.M(u)
        # print(P_pred)




        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):

        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        self.sigma_point(X_predict.reshape(-1, 1), P_predict, self.kappa_g)

        

        z1 = z[0:2]
        print("z1 = ", z1)
        # z2 = z.reshape(6, -1)[0:3, :]
        z2 = z[3:5]
        print("z2 = ", z2)
        # print("z.shape = ", z.shape)

        z_stack = np.concatenate((z1, z2), axis=0)
        
        z_pred = np.zeros_like(z_stack)

        for i in range(2 * self.n + 1):
            h1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.X[:, i])
            h2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.X[:, i])
            h_stack = np.concatenate((h1, h2), axis=0)
            z_pred += h_stack * self.w[i]

        z_pred /= (2 * self.n + 1)
        print("z_pred.shape = ", z_pred.shape)
        innovation = z_stack - z_pred

        Q_stack = block_diag(self.Q, self.Q)
        innovation_cov = np.zeros_like(Q_stack)
        cross = np.zeros((3, 4))

        for i in range(2 * self.n + 1):
            h1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.X[:, i])
            h2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.X[:, i])
            h_stack = np.concatenate((h1, h2), axis=0)
            # print(h_stack)
            innovation_cov += self.w[i] * (h_stack - z_pred).reshape(-1, 1) @ (h_stack - z_pred).reshape(-1, 1).T
            cross += self.w[i] * (self.X[:, i] - X_predict).reshape(-1, 1) @ (h_stack - z_pred).reshape(-1, 1).T
        
        innovation_cov += Q_stack

        K = cross @ np.linalg.inv(innovation_cov)
        X = X_predict + K @ innovation
        P = P_predict - K @ innovation_cov @ K.T

        # X = X_predict
        # P = P_predict


        # innovation_cov = H_stack @ P_predict @ H_stack.T + Q_stack

        # K = P_predict @ H_stack.T @ np.linalg.inv(innovation_cov)

        # X = X_predict + K @ innovation
        # P = (np.identity(3) - K @ H_stack) @ P_predict

        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)
        # print("shpae of self.w = ", self.w.shape)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state