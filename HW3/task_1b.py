import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time

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
        self.W = 0.01 * np.eye(3)
        self.V_1 = 0.0000000001 * 2.5 * self.getV(self.z_1)
        # print("self.V_1 = ", self.V_1)
        self.V_2 = 0.0000000001 * 2.5 * self.getV(self.z_2)
        # print("self.V_2 = ", self.V_2)
        self.V = block_diag(self.V_1, self.V_2)
        # print("self.V = ", self.V.shape)

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

    def H(self):
        H_1, H_2 = self.H_1(), self.H_2()
        return np.concatenate((H_1, H_2), axis=0)

    def h_1(self):
        return self.Kf_1 @ self.projection(self.p) + self.C_1

    def h_2(self):
        return self.Kf_2 @ self.projection(self.R.T @ self.p - self.R.T @ self.t) + self.C_2

    def h(self):
        h_1, h_2 = self.h_1(), self.h_2()
        return np.concatenate((h_1, h_2), axis=0)

    def predition(self):
        self.p = self.p
        self.sigma = self.sigma + self.W

    def correction(self, z):
        h = self.h()
        print("z = ", z, " h = ", h)
        innovation = z - h
        H = self.H()
        innovation_cov = H @ self.sigma @ H.T + self.V
        K = self.sigma @ H.T @ np.linalg.inv(innovation_cov)
        self.p = self.p + K @ innovation
        self.sigma = (np.eye(3) - K @ H) @ self.sigma

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
init.p = np.array([[4], [4], [1]])
init.sigma = 1000 * np.eye(3)


ekf = EKF(sys, init)
# print(ekf.H(), "\n", ekf.H_1(), ekf.H_2())
# print(ekf.h(), ekf.h_1(), ekf.h_2())

p = []
sigma = []
p.append(init.p)
sigma.append(init.sigma)
start = time.time()
for i in range(z_1.shape[0]):
    print(i)
    ekf.predition()
    z = np.concatenate((z_1[i].reshape((2, 1)), z_2[i].reshape((2, 1))), axis=0)
    ekf.correction(z)
    p.append(ekf.p)
    sigma.append(ekf.sigma)

end = time.time()

p = np.array(p)
print("p.shape = ", p.shape)
# plt.plot()
# print(len(p))
print(p[:, 0].shape)

plt.plot(p[:, 0], label='x')
plt.plot(p[:, 1], label='y')
plt.plot(p[:, 2], label='z')
plt.legend()
plt.title('Extended Kalman Filter batch version')
plt.xlabel('time step')
plt.ylabel('position')

plt.show()

print("Estimated object position = \n", p[-1, :])
print(f"Runtime of the program is {end - start}")