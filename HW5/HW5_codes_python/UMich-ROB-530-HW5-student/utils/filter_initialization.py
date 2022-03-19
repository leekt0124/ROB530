
import numpy as np
from functools import partial
from numpy.random import default_rng
rng = default_rng()

class myStruct:
    pass

init = myStruct()

#TODO: check all the conversions from Matlab: any typos?
def Gfun(mu,u):
    output = np.array([[1,0,(u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/u[1]],\
                        [0,1,(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]))/u[1]],\
                        [0,0,1]])
    return output

def Vfun(mu,u):
    output = np.array([[np.sin(mu[2]+u[1])/u[1]-np.sin(mu[2])/u[1], (u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]))/(u[1]**2), 0],\
                       [np.cos(mu[2])/u[1]-np.cos(mu[2]+u[1])/u[1], (u[0]*np.cos(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/(u[1]**2), 0],\
                       [0,                                          1,                                                                                             1]])

    return output

def Hfun(landmark_x, landmark_y, mu_pred, z_hat):
    output = np.array([
        [(landmark_y-mu_pred[1])/(z_hat[1]**2),   -(landmark_x-mu_pred[0])/(z_hat[1]**2),-1],\
        [-(landmark_x-mu_pred[0])/z_hat[1],       -(landmark_y-mu_pred[1])/z_hat[1],0]])
    return output
    


def filter_initialization(sys, initialStateMean, initialStateCov, filter_name):
    if filter_name == 'EKF':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        init.Gfun = Gfun
        init.Vfun = Vfun
        init.Hfun = Hfun
        from filter.EKF import EKF
        filter = EKF(sys, init)
    
    if filter_name == 'UKF':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        init.kappa_g = 2
        from filter.UKF import UKF
        filter = UKF(sys, init)

    if filter_name == 'PF':
        init.mu = initialStateMean
        init.Sigma = 0.001 * np.eye(3)
        init.n = 500
        init.particles = np.zeros((3, init.n))
        init.particle_weight = np.zeros(init.n)
        L = np.linalg.cholesky(init.Sigma) #TODO: check if the same effect
        for i in range(init.n):
            init.particles[:,i] = L @ rng.standard_normal((3,1)).reshape(3) + init.mu #TODO: check shape
            init.particle_weight[i] = 1/init.n
        from filter.PF import PF
        filter = PF(sys, init)

    if filter_name == "InEKF":
        init.mu = np.eye(3)
        init.mu[0,2] = initialStateMean[0]
        init.mu[1,2] = initialStateMean[1]
        init.Sigma = initialStateCov; # note: the covariance here is wrt lie algebra (not wrt Cartesian coordinate (x,y,theta))
        from filter.InEKF import InEKF
        filter = InEKF(sys, init)

    if filter_name == 'test':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        from filter.DummyFilter import DummyFilter
        filter = DummyFilter(sys, init)
    
    
    return filter