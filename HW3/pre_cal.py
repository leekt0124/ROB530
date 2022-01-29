import numpy as np

mat = scipy.io.loadmat('data.mat')

C_1 = mat["C_1"]    # (2, 1)
C_2 = mat["C_2"]    # (2, 1)
Kf_1 = mat["Kf_1"]  # (2, 2)
Kf_2 = mat["Kf_2"]  # (2, 2)
R = mat["R"]    # (3, 3)
t = mat["t"]    # (3, 1)
z_1 = mat["z_1"]    # (20, 2)
z_2 = mat["z_2"]    # (20, 2)

