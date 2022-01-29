import numpy as np
import scipy.io

# Read csv file
mat = scipy.io.loadmat('data.mat')

C_1 = mat["C_1"]
C_2 = mat["C_2"]
Kf_1 = mat["Kf_1"]
Kf_2 = mat["Kf_2"]
R = mat["R"]
t = mat["t"]
z_1 = mat["z_1"]
z_2 = mat["z_2"]

# def H_1(Kf_1, p):
#     px = p[0][0]
#     py = p[1][0]
#     pz = p[2][0]
#     temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
#     return Kf_1 @ temp

# def H_2(Kf_2, p, R):
#     px = p[0][0]
#     py = p[1][0]
#     pz = p[2][0]
#     temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
#     return Kf_2 @ temp @ R.T

# def h1(Kf_1, p, C_1):
#     return Kf_1 @ 

class EKF:
    def __init__(self, system, init):
        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance

    def projection(self, p):
        x = p[0][0]
        y = p[1][0]
        z = p[2][0]
        return np.array([[x / z], [y / z]])

    def H_1(self, Kf_1, p):
        px = p[0][0]
        py = p[1][0]
        pz = p[2][0]
        temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
        return Kf_1 @ temp

    def H_2(self, Kf_2, p, R):
        px = p[0][0]
        py = p[1][0]
        pz = p[2][0]
        temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
        return Kf_2 @ temp @ R.T

    def h1(self, Kf_1, p, C_1):
        return Kf_1 @ 

    def predition(self):

    def correction(self):