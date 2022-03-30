#!usr/bin/env python

#
# Generalized ICP over SE(3)
#
# Author: Fangtong Liu
# Date: 05/19/2020
#

import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import expm, logm


def sort_idxlist(idx, map, pt):
    map_idx = map[idx, :]
    dis_map = np.sqrt(np.sum((map_idx-pt)*(map_idx-pt), 1))
    dis_map_idx = np.concatenate((dis_map.reshape(-1, 1), np.array(idx).reshape(-1, 1)), 1)
    sorted_dis = dis_map_idx[np.lexsort(dis_map_idx[:, ::-1].T)]
    return sorted_dis[:, 1].astype(int)


def compute_jacobian(X, p_target, p_source, Ct, Cs, target_idx, survived_idx):
    A = np.zeros([6, 6])
    b = np.zeros([6, 1])
    R = X[0:3, 0:3]
    t = X[0:3, 3]
    # residual
    r = p_target - np.dot(p_source, R.T)
    r[:, 0] = r[:, 0] - t[0]
    r[:, 1] = r[:, 1] - t[1]
    r[:, 2] = r[:, 2] - t[2]
    n = r.shape[0]
    for i in range(n):
        # Inverse of covariance
        temp = np.linalg.cholesky(Ct[target_idx[i]] + np.dot(np.dot(R, Cs[survived_idx[i]]), R.T))
        invL = np.linalg.inv(temp)
        # Jacobian
        temp = np.hstack((skew(p_source[i, :]), -np.eye(3)))
        J = np.dot(invL, temp)
        # Left hand side matrix Ax=b
        A = A + np.dot(J.T, J)
        # Right hand side vector Ax=b
        b = b - np.dot(np.dot(J.T, invL), r[i, :]).reshape(-1, 1)
    return A, b



def pc_covariances_ann(pckdt, pcxyz):
    # Compute the empirical covariance at each point using ANN search
    e = 1e-2  # covariance epsilon
    C = []
    for i in range(pcxyz.shape[0]):
        dist, nn_id = pckdt.query(pcxyz[i, :], k=6)
        # HACK: adding a jitter to avoid singularity
        Cov = np.cov(pcxyz[nn_id, :].T) + e * np.eye(3)
        # GICP covariance
        D, V = np.linalg.eig(Cov)
        D = np.diag(D)
        D[0, 0] = e
        D[1, 1] = 1
        D[2, 2] = 1
        C.append(np.dot(np.dot(V, D), V.T))
    return C


def skew(x):
    # vector to skew R^3 -> so(3)
    X = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X


def hat(x):
    # hat: R^6 -> se(3)
    X = np.hstack((skew(x[0:3]), x[3:6].reshape(-1, 1)))
    X = np.vstack((X, np.array([0, 0, 0, 0])))
    return X


def gicp_SE3(target, source):
    # Generalized ICP over SE(3)

    # Create kd-tree objects for NN queries
    target_xyz = np.squeeze(target.astype(float))
    source_xyz = np.squeeze(source.astype(float))
    target_kdt = KDTree(target_xyz)
    source_kdt = KDTree(source_xyz)

    # Covariance normal at each point
    Ct = pc_covariances_ann(target_kdt, target_xyz)
    Cs = pc_covariances_ann(source_kdt, source_xyz)

    # initial guess
    T0 = np.eye(4)
    T1 = np.copy(T0)

    # ICP loop; find correspondences and optimize
    d_threshold = 1.5
    converged = False
    tf_epsilon = 1e-5
    iter = 0
    max_iter = 50
    inner_max_iter = 100
    eps_Jr = 1e-6
    while not converged and iter < max_iter:
        # apply the current transformation to the source point cloud
        current_source = np.dot(source_xyz, T0[0:3, 0:3].T)
        current_source[:, 0] = current_source[:, 0] + T0[0, 3]
        current_source[:, 1] = current_source[:, 1] + T0[1, 3]
        current_source[:, 2] = current_source[:, 2] + T0[2, 3]

        # NN queries
        dist, idx = target_kdt.query(current_source)

        # apply distance threshold to remove outliers
        survived_idx = np.where(dist < d_threshold)
        survived_idx = np.asarray(survived_idx).reshape(-1)
        target_idx = idx[survived_idx]

        p_source = source_xyz[survived_idx]
        p_target = target_xyz[target_idx]

        # solve for the new transformation
        # Gauss-Newton solver over SE(3)
        inner_iter = 0
        while inner_iter < inner_max_iter:
            inner_iter = inner_iter + 1
            # solve normal equations
            A, b = compute_jacobian(T1, p_target, p_source, Ct, Cs, target_idx, survived_idx)
            dx = np.dot(np.linalg.inv(A), b)

            # retract and update the estimate
            T1 = np.dot(expm(hat(dx)), T1)

            # if not np.mod(inner_iter, 5):
            #     print('GN Iter: ', inner_iter)

            # check if converged
            if np.linalg.norm(b) < eps_Jr:
                # if np.mod(inner_iter, 5):
                #     print('GN Iter: ', inner_iter)
                break

        # check if converged
        if np.linalg.norm(logm(np.dot(np.linalg.inv(T0), T1))) < tf_epsilon:
            print('Converged')
            converged = True
        else:
            T0 = np.copy(T1)
            iter = iter + 1
            # print('Iter: ', iter)
            if not (iter < max_iter):
                print('Not Converged. Maximum iteration of ', max_iter, ' is reached.')

    T = np.copy(T1)
    return T




