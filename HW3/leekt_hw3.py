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

class EKF:
    def __init__(self):

    def predition(self):

    def correction(self):