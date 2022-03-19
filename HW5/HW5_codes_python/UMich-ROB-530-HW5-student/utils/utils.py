import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

from system.RobotState import *

def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases

def func(x): # lie algebra -> cart
    return vee(expm(wedge(x)))

def unscented_propagate(mean, cov, kappa): # lie algebra
    n = np.size(mean)

    x_in = np.copy(mean)
    P_in = np.copy(cov)

    # sigma points
    L = np.sqrt(n + kappa) * np.linalg.cholesky(P_in)
    Y_temp = np.tile(x_in, (n, 1)).T
    X = np.copy(x_in.reshape(-1, 1))
    X = np.hstack((X, Y_temp + L))
    X = np.hstack((X, Y_temp - L))

    w = np.zeros((2 * n + 1, 1))
    w[0] = kappa / (n + kappa)
    w[1:] = 1 / (2 * (n + kappa))

    new_mean = np.zeros((n, 1))
    new_cov = np.zeros((n, n))
    Y = np.zeros((n, 2 * n + 1))
    for j in range(2 * n + 1):
        Y[:,j] = func(X[:,j])
        new_mean[:,0] = new_mean[:,0] + w[j] * Y[:,j]

    diff = Y - new_mean
    for j in range(np.shape(diff)[1]):
        diff[2,j] = wrap2Pi(diff[2,j])

    w_mat = np.zeros((np.shape(w)[0],np.shape(w)[0]))
    np.fill_diagonal(w_mat,w)
    new_cov = diff @ w_mat @ diff.T
    cov_xy = (X - x_in.reshape(-1,1)) @ w_mat @ diff.T

    return new_cov

def wedge(x):
    # wedge operation for so(3) to put an R^3 vector to the Lie algebra basis
    G1 = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])  
    G2 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0]])  
    G3 = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])  
    xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
    return xhat

def vee(x):
    return np.array([x[0, 2], x[1, 2], x[1, 0]])

def pose_mat(X):
    x = X[0]
    y = X[1]
    h = X[2]
    H = np.array([[np.cos(h),-np.sin(h),x],\
                [np.sin(h),np.cos(h),y],\
                [0,0,1]])
    return H


def lieToCartesian(mean, cov): # mean: lie group, cov: lie algebra
    ###############################################################################
    # TODO: Implement the lieToCartesian function for extra credits               #
    # Hint: you can use unscented transform                                       #
    # Hint: save the mean and cov as mu_cart and Sigma_cart                       #
    ###############################################################################
    #                         END OF YOUR CODE                                    #
    ###############################################################################
    # se(2) -> SE(2)
    mean_H = pose_mat(mean)
    mean_se2 = vee(logm(mean_H))
    mu_cart = mean
    Sigma_cart = unscented_propagate(mean_se2, cov, 2)

    print("mean = ", mean)
    print("cov = ", cov)
    print("mean_H = ", mean_H)
    print("mean_se2 = ", mean_se2)

    return mu_cart, Sigma_cart

def mahalanobis(state, ground_truth, filter_name, Lie2Cart):
    # Output format (7D vector) : 1. Mahalanobis distance
    #                             2-4. difference between groundtruth and filter estimation in x,y,theta 
    #                             5-7. 3*sigma (square root of variance) value of x,y, theta 
    
    ###############################################################################
    # TODO: Implement the mahalanobis function for extra credits                  #
    # Output format (7D vector) : 1. Mahalanobis distance                         #
    #    2-4. difference between groundtruth and filter estimation in x,y,theta   #
    #    5-7. 3*sigma (square root of variance) value of x,y, theta               #
    # Hint: you can use state.getState() to get mu, state.getCovariance() to      #
    #       get covariance for EKF, UKF, PF.                                      #
    #       For InEKF, if Lie2Cart flag is true, you should use                   #
    #       state.getCartesianState() and state.getCartesianCovariance() instead. #
    ###############################################################################
    results = np.zeros((7, 1))
    if filter_name in ["EKF", "PF", "UKF"]:
        cov = state.getCovariance()
        X = state.getState()
    elif filter_name == "InEKF":
        cov = state.getCartesianCovariance()
        X = state.getCartesianState()
        

    offset = (X - ground_truth).reshape(-1, 1)
    results[1:4, :] = offset
    dis = np.sqrt(offset.T @ np.linalg.inv(cov) @ offset)
    results[0, :] = dis
    results[4] = np.sqrt(cov[0, 0]) * 3
    results[5] = np.sqrt(cov[1, 1]) * 3
    results[6] = np.sqrt(cov[2, 2]) * 3
    print("cov = ", cov)
    print("offset = ", offset)
    print("results = ", results)


    
    ###############################################################################
    #                         END OF YOUR CODE                                    #
    ###############################################################################
    return results.reshape(-1)
    

def plot_error(results, gt):
    num_data_range = range(np.shape(results)[0])

    gt_x = gt[:,0]
    gt_y = gt[:,1]

    plot2 = plt.figure(2)
    plt.plot(num_data_range,results[:,0])
    plt.plot(num_data_range, 7.81*np.ones(np.shape(results)[0]))
    plt.title("Chi-square Statistics")
    plt.legend(["Chi-square Statistics", "p = 0.05 in 3 DOF"])
    plt.xlabel("Iterations")

    plot3,  (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title("Deviation from Ground Truth with 3rd Sigma Contour")
    ax1.plot(num_data_range, results[:,1])
    ax1.set_ylabel("X")
    ax1.plot(num_data_range,results[:,4],'r')
    ax1.plot(num_data_range,-1*results[:,4],'r')
    
    ax1.legend(["Deviation from Ground Truth","3rd Sigma Contour"])
    ax2.plot(num_data_range,results[:,2])
    ax2.plot(num_data_range,results[:,5],'r')
    ax2.plot(num_data_range,-1*results[:,5],'r')
    ax2.set_ylabel("Y")

    ax3.plot(num_data_range,results[:,3])
    ax3.plot(num_data_range,results[:,6],'r')
    ax3.plot(num_data_range,-1*results[:,6],'r')
    ax3.set_ylabel("theta")
    ax3.set_xlabel("Iterations")
    
    plt.show()


def main():

    i = -7
    j = wrap2Pi(i)
    print(j)
    

if __name__ == '__main__':
    main()