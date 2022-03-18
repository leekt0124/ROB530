
from mimetypes import init
from os import stat
from stat import UF_APPEND
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        self.A = np.zeros((3, 3))
        self.dt = 1

        self.mu = init.mu # (3, 3)
        self.Sigma = init.Sigma # (3, 3)

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])]) # (3, )
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    def Ad(self, X):
        # Adjoint in SO(3)
        # See http://ethaneade.com/lie.pdf for detail derivation
        # x = deepcopy(X[0, 2])
        # y = deepcopy(X[1, 2])
        # X[0, 2] = y
        # X[1, 2] = -x
        # print("X = ", X)
        # return X
        AdX = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
        AdX = np.vstack((AdX, np.array([0, 0, 1])))
        # print("Adx = ", AdX)
        return AdX

    def wedge(self, x):
        # wedge operation for so(3) to put an R^3 vector to the Lie algebra basis
        G1 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]])  
        G2 = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 0, 0]])  
        G3 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])  
        xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
        return xhat

    
    def prediction(self, u):
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        print("self.mu = ", self.mu)
        adjX = self.Ad(self.mu)
        print("adjX = ", adjX)
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.propagation(u_se2, adjX)

    def propagation(self, u, adjX):
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################
        # phi = expm(self.A)
        # self.P_pred = phi @ self.Sigma @ phi.T + adjX @ self.W @ adjX.T
        self.P_pred = self.Sigma + adjX @ self.W @ adjX.T
        self.X_pred = self.mu @ expm(u)
        print("P_pred = ", self.P_pred)
        print("X_pred = ", self.X_pred)
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
    def H(self, m):
        return np.array([[-1, 0, m[1]], [0, -1, -m[0]], [0, 0, 0]])

    
    def correction(self, Y1, Y2, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        V_stack = block_diag(self.V, np.zeros((1, 1)))
        # print("V_stack = ", V_stack)

        N = (self.X_pred @ V_stack @ self.X_pred.T)[0:2, 0:2]
        N_stack = block_diag(N, N)
        print("N_stack = ", N_stack)

        m1x = landmark1.getPosition()[0]
        m1y = landmark1.getPosition()[1]
        m1 = np.array([m1x, m1y, 1])
        print("m1 = ", m1)
    
        m2x = landmark2.getPosition()[0]
        m2y = landmark2.getPosition()[1]
        m2 = np.array([m2x, m2y, 1])
        print("m2 = ", m2)



        H1 = self.H(m1)[0:2, :]
        H2 = self.H(m2)[0:2, :]

        H = np.vstack((H1, H2))
        # print("H = ", H)

        S = H @ self.P_pred @ H.T + N_stack
        L = self.P_pred @ H.T @ np.linalg.inv(S)

        print("Y1 = ", Y1)
        print("self.X_pred @ Y1 = ", self.X_pred @ Y1)
        print("self.X_pred @ Y2 = ", self.X_pred @ Y2)
        XY1 = (self.X_pred @ Y1).reshape(-1, 1)[0:2, :]
        XY2 = (self.X_pred @ Y2).reshape(-1, 1)[0:2, :]
        XY = np.vstack((XY1, XY2))
        print("XY = ", XY)

        b = np.vstack((m1.reshape(-1, 1)[0:2, :], m2.reshape(-1, 1)[0:2, :]))
        print("b = ", b)





        self.mu = expm(self.wedge(L @ (XY - b))) @ self.X_pred
        self.Sigma = (np.eye(3) - L @ H) @ self.P_pred @ (np.eye(3) - L @ H).T + L @ N_stack @ L.T

        print("correction")
        # self.mu = self.X_pred
        # self.Sigma = self.P_pred

        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])]).reshape(-1)
        print("X = ", X)
        print("self.Sigma = ", self.Sigma)

        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
