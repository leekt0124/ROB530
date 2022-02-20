import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation as R


class Right_IEKF:
    
    def __init__(self, system):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        self.A = system['A']  # error dynamics matrix
        self.f = system['f']  # process model
        self.H = system['H']  # measurement error matrix
        self.Q = system['Q']  # input noise covariance
        self.V = system['V']  # measurement noise covariance
        self.X = system['init_X']    # state vector, so(3)
        self.P = system['init_P']  # state covariance, adjustable

    def Ad(self, X):
        # Adjoint in SO(3)
        # See http://ethaneade.com/lie.pdf for detail derivation
        return X

    def wedge(self, x):
        # wedge operation for so(3) to put an R^3 vector to the Lie algebra basis
        G1 = np.array([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])  # omega_1
        G2 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]])  # omega_2
        G3 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])  # omega_3
        xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
        return xhat

    def prediction(self, u, dt):
        phi = expm(self.A)
        self.P = phi @ self.P @ phi.T + self.Ad(self.X) @ self.Q @ self.Ad(self.X).T
        self.X = self.f(self.X, self.wedge(u) * dt)

    def correction(self, Y, b):
        N = self.X @ self.V @ self.X.T
        S = self.H(b) @ self.P @ self.H(b).T + N
        L = self.P @ self.H(b).T @ np.linalg.inv(S)

        # Update state
        nu = self.X @ Y - b
        delta = self.wedge(L @ nu)
        self.X = expm(delta) @ self.X

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - L @ self.H(b)
        self.P = temp @ self.P @ temp.T + L @ N @ L.T

def motion_model(X, u_dt):
    return X @ expm(u_dt)

def measurement_error_matrix(m):
    H = -np.array([[0, m[2][0], -m[1][0]], [-m[2][0], 0, m[0][0]], [m[1][0], -m[0][0], 0]])
    return H

system = {}
system['A'] = np.zeros((3, 3))
system['f'] = motion_model
system['H'] = measurement_error_matrix
system['Q'] = np.diag(np.power([0.015, 0.01, 0.01], 2))
system['V'] = np.diag(np.power([100 * 0.5, 100 * 0.5, 100 * 0.5], 2))
system['init_X'] = np.eye(3)
system['init_P'] = 0.1 * np.eye(3)

# Read data form mat file
mat = scipy.io.loadmat('data.mat')
a = mat["a"]
dt = mat["dt"]
euler_gt = mat["euler_gt"]
g = mat["g"]
omega = mat["omega"]

# Create riekf object
riekf = Right_IEKF(system)
R_data = [system['init_X']]

# Iteratively run right invariant EKF
for i in range(len(dt)):
    riekf.prediction(omega[i].reshape(-1, 1), dt[i][0])
    riekf.correction(a[i].reshape(-1, 1), g)
    R_data.append(riekf.X)

roll = []
pitch = []
yaw = []
for i in range(len(R_data)):
    r = R.from_matrix(R_data[i])
    euler = r.as_euler('zyx', degrees=False)
    yaw.append(euler[0])
    pitch.append(euler[1])
    roll.append(euler[2])

# Transform dt to t
t = np.cumsum(dt)
t = np.insert(t, 0, 0)

# Plot roll
plt.plot(t, roll, label='roll')
plt.plot(t, euler_gt[:, 2], label='gt_roll')
plt.title("Roll")
plt.ylabel("roll")
plt.xlabel("t")
plt.legend()
plt.show()

# Plot pitch
plt.plot(t, pitch, label='pitch')
plt.plot(t, euler_gt[:, 1], label='gt_pitch')
plt.title("Pitch")
plt.ylabel("pitch")
plt.xlabel("t")
plt.legend()
plt.show()

# Plot yaw
plt.plot(t, yaw, label='yaw')
plt.plot(t, euler_gt[:, 0], label='gt_yaw')
plt.title("Yaw")
plt.ylabel("yaw")
plt.xlabel("t")
plt.legend()
plt.show()