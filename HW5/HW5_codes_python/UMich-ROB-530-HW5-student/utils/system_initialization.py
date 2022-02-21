
import numpy as np
from functools import partial

from utils.utils import wrap2Pi

class myStruct:
    pass

def gfun(mu, u):
    output = np.zeros(3)
    output[0] = mu[0] + (-u[0] / u[1] * np.sin(mu[2]) + u[0] / u[1] * np.sin(mu[2] + u[1]))
    output[1] = mu[1] + ( u[0] / u[1] * np.cos(mu[2]) - u[0] / u[1] * np.cos(mu[2] + u[1]))
    output[2] = mu[2] + u[1] + u[2]
    return output

def hfun(landmark_x, landmark_y, mu_pred):
    output = np.zeros(2)
    output[0] = wrap2Pi(np.arctan2(landmark_y - mu_pred[1], landmark_x - mu_pred[0]) - mu_pred[2])
    output[1] = np.sqrt((landmark_y - mu_pred[1])**2 + (landmark_x - mu_pred[0])**2)
    return output

def M(u, alphas):
    output = np.array([[alphas[0]*u[0]**2+alphas[1]*u[1]**2,0,0], \
        [0,alphas[2]*u[0]**2+alphas[3]*u[1]**2,0],\
            [0,0,alphas[4]*u[0]**2+alphas[5]*u[1]**2]])
    return output

def system_initialization(alphas, beta):
    sys = myStruct()

    sys.gfun = gfun
    sys.hfun = hfun
    sys.M = partial(M, alphas=alphas)
    sys.Q = np.array([[beta**2,0],[0,25**2]])
    sys.W = np.diag([0.05**2,0.1**2,0.1**2])
    sys.V = 10000*np.eye(2)
    
    return sys

