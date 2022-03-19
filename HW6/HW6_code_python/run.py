import sys
import argparse
import numpy as np
from scipy.io import loadmat
import os
from utils import plot_mean, plot_variance, plot_semantic
from ogm_CSM import ogm_CSM
from ogm_continuous_CSM import ogm_continuous_CSM
from ogm_S_CSM import ogm_S_CSM
from ogm_continous_S_CSM import ogm_continous_S_CSM


def parse_args():
    parser = argparse.ArgumentParser(description='Run HW6')
    parser.add_argument('--task_num', type=int, help='Which task are you running?',
                        default=1)
    parser.add_argument('--plot', dest='plot', action='store_false')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print('Running task %d' % args.task_num)

    # load data
    if args.task_num == 1 or args.task_num == 2:
        dataLoad = loadmat(os.path.join('data','sample_Intel_dataset.mat'))
        robotPose = dataLoad['robotPose']
        # print(robotPose['x'][0][0].shape) % (743, 1)
        # print(robotPose['h'][0][0]) % (743, 1)
        laserScan = dataLoad['laserScan']
        # print(laserScan[0][0].shape) # (2, 163)
    elif args.task_num == 3 or args.task_num == 4:
        dataLoad = loadmat(os.path.join('data','sample_Intel_dataset_semantic.mat'))
        robotPose = dataLoad['robotPose']
        laserScan = dataLoad['laserScan']
    else:
        print('wrong input of task_num')   

    # run tasks
    if args.task_num == 1: # task 1, ogm_CSM
        # initialize map
        ogm = ogm_CSM()

        # build map
        ogm.construct_map(robotPose, laserScan)
        ogm.build_ogm()

        # plot
        if args.plot:
            plot_mean(ogm, 'CSM Mean', 'ogm_intel_CSM_mean.png')
            plot_variance(ogm, 'CSM Variance', 'ogm_intel_CSM_variance.png')

    elif args.task_num == 2: # task 2, ogm_continuous_CSM
        # initialize map
        ogm = ogm_continuous_CSM()
        
        # build map
        ogm.construct_map(robotPose, laserScan)
        ogm.build_ogm()

        # plot
        if args.plot:
            plot_mean(ogm, 'Continuous CSM Mean', 'ogm_intel_continuous_CSM_mean.png')
            plot_variance(ogm, 'Continuous CSM Variance', 'ogm_intel_continuous_CSM_variance.png')

    elif args.task_num == 3: # task 3, ogm_S_CSM
        # initialize map
        ogm = ogm_S_CSM()
        
        # build map
        ogm.construct_map(robotPose, laserScan)
        ogm.build_ogm()

        # plot
        if args.plot:
            plot_semantic(ogm, 'S-CSM Mean', 'ogm_intel_S_CSM_mean.png')
            plot_variance(ogm, 'S-CSM Variance', 'ogm_intel_S_CSM_variance.png')

    elif args.task_num == 4: # task 4, ogm_continous_S_CSM
        # initialize map
        ogm = ogm_continous_S_CSM()
        
        # build map
        ogm.construct_map(robotPose, laserScan)
        ogm.build_ogm()

        # plot
        if args.plot:
            plot_semantic(ogm, 'Continuous S-CSM Mean', 'ogm_intel_continuous_S_CSM_mean.png')
            plot_variance(ogm, 'Continuous S-CSM Variance (grid size = 0.135)', 'ogm_intel_continuous_S_CSM_variance_grid_0135.png')

    else:
        print('wrong input of task_num')


if __name__ == '__main__':
    main()
