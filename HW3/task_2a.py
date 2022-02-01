import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from numpy.random import randn
from scipy.stats import multivariate_normal
from numpy.random import randn, rand

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

class Particle():
    def __init__(self):
        self.p = []  # particle state 
        self.w = []  # importance weights

class PF:
    def __init__(self, system, init):
        self.C_1 = system.C_1
        self.C_2 = system.C_2
        self.Kf_1 = system.Kf_1
        self.Kf_2 = system.Kf_2
        self.R = system.R
        self.t = system.t
        self.z_1 = system.z_1
        self.z_2 = system.z_2

        self.p = init.p
        self.sigma = init.sigma
        self.n = init.n
        self.particle = Particle()

        self.W = system.W
        self.V_1 = system.V_1
        self.V_2 = system.V_2

        self.LW = np.linalg.cholesky(self.W)  # Cholesky factor of Q

        wu = 1 / self.n  # uniform weights = 0.01
        # print(wu)
        L_init = np.linalg.cholesky(self.sigma)
        for i in range(self.n):
            self.particle.p.append(np.dot(L_init, randn(len(init.p), 1)) + init.p)
            self.particle.w.append(wu)
        self.particle.p = np.array(self.particle.p).reshape(-1, len(init.p))
        self.particle.w = np.array(self.particle.w).reshape(-1, 1)
        # print(self.particle.w)

    def getV(self, z):
        return np.cov(z[:, 0], z[:, 1])

    def projection(self, p):
        x = p[0][0]
        y = p[1][0]
        z = p[2][0]
        return np.array([[x / z], [y / z]])

    def h_1(self, p):
        return self.Kf_1 @ self.projection(p) + self.C_1

    def h_2(self, p):
        return self.Kf_2 @ self.projection(self.R.T @ p - self.R.T @ self.t) + self.C_2

    def motion(self, p, noise):
        return p + noise

    def prediction(self):
        for i in range(self.n):
            # sample noise
            noise = (self.LW @ randn(3, 1)).reshape(-1)
            self.particle.p[i, :] = self.motion(self.particle.p[i, :], noise)

    def correction_1(self, z):
        # Update only one sensor
        prob = np.zeros((self.n, 1))
        # Update z
        for i in range(self.n):
            # print(z)
            # print(self.h_1(self.particle.p[i, :].reshape((3, 1))))
            innovation = z - self.h_1(self.particle.p[i, :].reshape((3, 1)))
            # print("innovation = ", innovation)
            prob[i][0] = multivariate_normal.pdf(innovation.reshape(-1), mean=np.array([0, 0]), cov=self.V_1)
            # print(prob.shape)
            # print("self.V_1 = ", self.V_1)
            # print("innovation = ", innovation.reshape(-1))
            # print("prob = ", multivariate_normal.pdf(innovation.reshape(-1), mean=np.array([0, 0]), cov=self.V_1))

        # print(prob[i][0])

        self.particle.w = np.multiply(self.particle.w, prob)
        self.particle.w = self.particle.w / np.sum(self.particle.w)
        # print(self.particle.w)
        self.Neff = 1 / np.sum(np.power(self.particle.w, 2))
        print("self.Neff = ", self.Neff)


    def correction_2(self, z):
        # Update only one sensor
        prob = np.zeros((self.n, 1))
        # Update z
        for i in range(self.n):
            # print(z)
            # print(self.h_1(self.particle.p[i, :].reshape((3, 1))))
            innovation = z - self.h_2(self.particle.p[i, :].reshape((3, 1)))
            # print("innovation = ", innovation)
            prob[i][0] = multivariate_normal.pdf(innovation.reshape(-1), mean=np.array([0, 0]), cov=self.V_2)
            # print(prob.shape)
            # print("self.V_1 = ", self.V_1)
            # print("innovation = ", innovation.reshape(-1))
            # print("prob = ", multivariate_normal.pdf(innovation.reshape(-1), mean=np.array([0, 0]), cov=self.V_1))

        # print(prob[i][0])

        self.particle.w = np.multiply(self.particle.w, prob)
        self.particle.w = self.particle.w / np.sum(self.particle.w)
        # print(self.particle.w)
        self.Neff = 1 / np.sum(np.power(self.particle.w, 2))
        print("self.Neff = ", self.Neff)

    def resampling(self):
        W = np.cumsum(self.particle.w)
        r = rand(1) / self.n
        j = 1
        for i in range(self.n):
            u = r + (i - 1) / self.n
            while u > W[j]:
                j += 1
            self.particle.p[i, :] = self.particle.p[j, :]
            self.particle.w[i] = 1 / self.n


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
sys.W = 0.01 * np.eye(3)
sys.V_1 = np.array([[18.0, -2.92659222], [-2.92659222, 35.5]])
sys.V_2 = np.array([[144.5, 4.83429415], [4.83429415, 67.8960584]])

init = myStruc()
init.n = 200
init.p = np.array([[1], [1], [1]])
init.sigma = 1 * np.eye(3)

pf = PF(sys, init)
# pf.prediction()
p = list()
p.append(init.p)
print(init.p)

for i in range(z_1.shape[0]):
    print(i)
    pf.prediction()
    pf.correction_1(z_1[i].reshape((2, 1)))
    
    if pf.Neff < pf.n / 5:
        pf.resampling()
    pf.correction_2(z_2[i].reshape((2, 1)))

    if pf.Neff < pf.n / 5:
        pf.resampling()

    x = np.mean(pf.particle.p[:, 0])
    y = np.mean(pf.particle.p[:, 1])
    z = np.mean(pf.particle.p[:, 2])
    p.append(np.array([[x], [y], [z]]))

p = np.array(p)

plt.plot(p[:, 0], label='x')
plt.plot(p[:, 1], label='y')
plt.plot(p[:, 2], label='z')
plt.legend()
plt.title('Particle Filter sequential version')
plt.xlabel('time step')
plt.ylabel('position')

plt.show()

print("Estimated object position = \n", p[-1, :])