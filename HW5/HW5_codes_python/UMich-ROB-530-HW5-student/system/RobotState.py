import numpy as np
import rospy

from scipy.spatial.transform import Rotation

import yaml

from utils.utils import *

class RobotState:
    '''
    Robot State:

    2D: X = [x,y,theta]; position = [x,y]; orientation = [theta]
    3D: X = [R t; 0 1]; position = [x,y,z]; orientation = R

    '''
    def __init__(self, time_stamp=None, position=None, orientation=None):
        
        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        self.world_dim = param["world_dimension"]
        self.filter_name = param["filter_name"]
        self.Lie2Cart = param['Lie2Cart']

        if time_stamp is not None:
            self.time_stamp_ = time_stamp
        else:
            self.time_stamp_ = 0
        
        if self.world_dim == 2:
            self.position_ = np.zeros(2)
            self.orientation_ = 0    # Orientation of the robot. If 2D then orientation = theta.
            self.X_ = np.zeros(3)
            self.P_ = np.eye(3)
        elif self.world_dim == 3:
            self.position_ = np.zeros(3)
            self.orientation_ = np.zeros(4) # If 3D then orientation is represented using quaternion.
            self.X_ = np.zeros((4,4))   # if 3D X = [R t; 0 1]
            self.P_ = np.zeros((6,6))
        

        if position is not None:
            self.setPosition(position)

        if orientation is not None:
            self.setOrientation(orientation)
    
    def setTime(self, time_stamp):
        if time_stamp is not None and isinstance(id,float):
            self.time_stamp_ = time_stamp

    def getTime(self):
        return self.time_stamp_

    def setPosition(self, position):
        if position is not None and isinstance(position, np.ndarray):
            self.position_ = np.copy(position)
            if self.world_dim == 2:
                self.X_[0:2] = self.position_
            elif self.world_dim == 3:
                self.X_[0:3] = self.position_
        elif position is not None and isinstance(position, list):
            self.position_ = np.array(position)
            if self.world_dim == 2:
                self.X_[0:2] = self.position_
            elif self.world_dim == 3:
                self.X_[0:3] = self.position_
        else:
            print("robot position is not set!")

        
    def getPosition(self):
        return np.copy(self.position_)

    def setOrientation(self, orientation):
        if orientation is not None and isinstance(orientation, np.ndarray):
            self.orientation_ = np.copy(orientation)
            if self.world_dim == 2:
                self.X_[2] = self.orientation_
            elif self.world_dim == 3:
                self.X_[0:3,0:3] = Rotation.from_quat(self.orientation_).as_matrix()

    def getOrientation(self):
        return np.copy(self.orientation_)


    def setPositionCovariance(self, cov_in):
        if self.world_dim == 2:
            self.P_[0:2,0:2] = cov_in
        elif self.world_dim == 3:
            self.P_[3:6,3:6] = cov_in


    def getPositionCovariance(self):
        if self.world_dim == 2:
            return np.copy(self.P_[0:2,0:2])
        elif self.world_dim == 3:
            return np.copy(self.P_[3:6,3:6])


    def setOrientationCovariance(self, cov_in):
        if self.world_dim == 2:
            self.P_[3,3] = cov_in
        elif self.world_dim == 3:
            self.P_[0:3,0:3] = cov_in


    def getOrientationCovariance(self):
        if self.world_dim == 2:
            return np.copy(self.P_[3,3])
        elif self.world_dim == 3:
            return np.copy(self.P_[0:3,0:3])

    def setCovariance(self, cov_in):
        self.P_ = cov_in

    def getCovariance(self):
        return np.copy(self.P_)

    def getCartesianCovariance(self):
        
        if self.filter_name != "InEKF":
            return np.copy(self.P_)
        elif self.Lie2Cart:
            self.mu_cart, self.P_cart = lieToCartesian(self.X_, self.P_)
            return np.copy(self.P_cart)
        else:
            print("Lie to Cartesian disabled. Returning zero cov.")
            return np.zeros((np.shape(self.X_)[0],np.shape(self.X_)[0]))

    def setState(self, X_in):
        self.X_ = X_in
        
        if self.world_dim == 2:
            self.position_ = self.X_[0:2]
            self.orientation_ = self.X_[2]
        elif self.world_dim == 3:
            self.position_ = self.X_[0:3] 
            self.orientation_ = Rotation.from_matrix(self.X_[0:3,0:3]).as_quat()

    def getState(self):
        return np.copy(self.X_)

    def getCartesianState(self):
        return np.copy(self.mu_cart)

def main():

    rob_sys = RobotState()

    pass

if __name__ == '__main__':
    main()