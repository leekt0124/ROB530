import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model (state transform)
        self.Vfun = init.Vfun  # Jocabian of motion model (control signal)
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        print("X = ", X)
        print("P = ", P)
        X_pred = self.gfun(X, u)
        print("X_pred = ", X_pred)
        # print("self.Gfun = ", self.Gfun)
        # print("self.Vfun = ", self.Vfun)
        G = self.Gfun(X, u)
        V = self.Vfun(X, u)
        # alphas = [0.00025, 0.00005, 0.0025, 0.0005, 0.0025, 0.0005]
        print("G = ", G)
        print("V = ", V)
        print("self.M = ", self.M(u))
        M = self.M(u)
        P_pred = G @ P @ G.T + V @ M @ V.T
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
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
        # z.reshape((1, -1))
        # np.reshape(z, (6, -1))
        # print(type(z))
        # z.reshape(2, 3)
        # print(z)
        # z1 = z.reshape(6, -1)[0:3, :]
        z1 = z[0:2]
        print("z1 = ", z1)
        # z2 = z.reshape(6, -1)[0:3, :]
        z2 = z[3:5]
        print("z2 = ", z2)
        # print("z.shape = ", z.shape)

        z_stack = np.concatenate((z1, z2), axis=0)
        
        h1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict)
        H1 = self.Hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict, z1)

        h2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict)
        H2 = self.Hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict, z2)

        h_stack = np.concatenate((h1, h2), axis=0)
        H_stack = np.concatenate((H1, H2), axis=0)

        innovation = z_stack - h_stack

        Q_stack = block_diag(self.Q, self.Q)

        innovation_cov = H_stack @ P_predict @ H_stack.T + Q_stack

        K = P_predict @ H_stack.T @ np.linalg.inv(innovation_cov)

        X = X_predict + K @ innovation
        P = (np.identity(3) - K @ H_stack) @ P_predict



        # innovation = z1 - h
        # print("P_predict = ", P_predict)
        # print("H = ", H)
        # print("Q = ", self.Q)
        # innovation_cov = H @ P_predict @ H.T + self.Q
        # print("innovation_cov = ", innovation_cov)
        # K = P_predict @ H.T @ np.linalg.inv(innovation_cov)
        # X_predict = X_predict + K @ innovation
        # P_predict = (np.identity(3) - K @ H) @ P_predict

        # h = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict)
        # H = self.Hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict, z2)
        # innovation = z2 - h
        # innovation_cov = H @ P_predict @ H.T + self.Q
        # K = P_predict @ H.T @ np.linalg.inv(innovation_cov)
        # X = X_predict + K @ innovation
        # P = (np.identity(3) - K @ H) @ P_predict
    

        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state