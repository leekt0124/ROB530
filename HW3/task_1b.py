import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Read csv file
mat = scipy.io.loadmat('data.mat')

C_1 = mat["C_1"]    # (2, 1)
C_2 = mat["C_2"]    # (2, 1)
Kf_1 = mat["Kf_1"]  # (2, 2)
Kf_2 = mat["Kf_2"]  # (2, 2)
R = mat["R"]    # (3, 3)
t = mat["t"]    # (3, 1)
z_1 = mat["z_1"]    # (20, 2)
z_2 = mat["z_2"]    # (20, 2)

class EKF:
    def __init__(self, system, init):
        self.C_1 = system.C_1
        self.C_2 = system.C_2
        self.Kf_1 = system.Kf_1
        self.Kf_2 = system.Kf_2
        self.R = system.R
        self.t = system.t
        self.z_1 = system.z_1
        self.z_2 = system.z_2

        # Need to be tuned
        self.W = 0.1 * np.eye(3)
        self.V_1 = 2.5 * self.getV(self.z_1)
        print("self.V_1 = ", self.V_1)
        self.V_2 = 2.5 * self.getV(self.z_2)
        print("self.V_2 = ", self.V_2)

        self.p = init.p  # state vector
        self.sigma = init.sigma  # state covariance

    def getV(self, z):
        return np.cov(z[:, 0], z[:, 1])

    def projection(self, p):
        x = p[0][0]
        y = p[1][0]
        z = p[2][0]
        return np.array([[x / z], [y / z]])

    def H_1(self):
        px = self.p[0][0]
        py = self.p[1][0]
        pz = self.p[2][0]
        temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
        return self.Kf_1 @ temp

    def H_2(self):
        px = self.p[0][0]
        py = self.p[1][0]
        pz = self.p[2][0]
        temp = np.array([[1/pz, 0, -px / pz ** 2], [0, 1/pz, -py / pz ** 2]])
        return self.Kf_2 @ temp @ self.R.T

    def h_1(self):
        return self.Kf_1 @ self.projection(self.p) + self.C_1

    def h_2(self):
        return self.Kf_2 @ self.projection(self.R.T @ self.p - self.R.T @ self.t) + self.C_2

    def predition(self):
        self.p = self.p
        self.sigma = self.sigma + self.W

    def correction(self, z_1, z_2):
        # correct z1
        # print("1 ", self.p)
        h_1 = self.h_1()
        print("z_1 = ", z_1, " h_1 = ", h_1)
        innovation = z_1 - h_1
        H_1 = self.H_1()
        innovation_cov = H_1 @ self.sigma @ H_1.T + self.V_1
        K = self.sigma @ H_1.T @ np.linalg.inv(innovation_cov)
        self.p = self.p + K @ innovation
        self.sigma = (np.eye(3) - K @ H_1) @ self.sigma
        # self.sigma = (np.eye(3) - K @ H_1) @ self.sigma @ (np.eye(3) - K @ H_1).T + K @ self.V_1 @ K.T

        # correct z2
        h_2 = self.h_2()
        # print("2 ", self.p)
        print("z_2 = ", z_2, " h_2 = ", h_2)
        innovation = z_2 - h_2
        H_2 = self.H_2()
        innovation_cov = H_2 @ self.sigma @ H_2.T + self.V_2
        K = self.sigma @ H_2.T @ np.linalg.inv(innovation_cov)
        self.p = self.p + K @ innovation
        print(self.p)
        # print(self.sigma)
        self.sigma = (np.eye(3) - K @ H_2) @ self.sigma
        # self.sigma = (np.eye(3) - K @ H_2) @ self.sigma @ (np.eye(3) - K @ H_2).T + K @ self.V_2 @ K.T

class myStruc:
    pass

sys = myStruc()
sys.C_1 = C_1
sys.C_2 = C_2
sys.Kf_1 = Kf_1
sys.Kf_2 = Kf_2
sys.R = R
sys.t = t
sys.z_1 = z_1
sys.z_2 = z_2


init = myStruc()
init.p = np.array([[1], [1], [1]])
init.sigma = 1000 * np.eye(3)


ekf = EKF(sys, init)
p = []
sigma = []
p.append(init.p)
sigma.append(init.sigma)
for i in range(z_1.shape[0]):
    print(i)
    ekf.predition()
    ekf.correction(z_1[i].reshape((2, 1)), z_2[i].reshape((2, 1)))
    p.append(ekf.p)
    sigma.append(ekf.sigma)

p = np.array(p)
# plt.plot()
# print(len(p))
print(p[:, 0].shape)
plt.plot(p[:, 0], label='x')
plt.plot(p[:, 1], label='y')
plt.plot(p[:, 2], label='z')
plt.legend()
plt.show()