import sys
sys.path.append('.')
import yaml
import matplotlib.pyplot as plt


import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

from system.RobotState import *
from comm.path_publisher import *
from comm.marker_publisher import *
from utils.DataHandler import *
from utils.filter_initialization import filter_initialization
from utils.system_initialization import system_initialization
from utils.utils import *

class RobotSystem:
    

    def __init__(self, world=None):

        rospy.init_node('robot_state_estimator', anonymous=True)

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        
        # load motion noise and sensor noise
        alphas = np.array(param['alphas_sqrt'])**2
        beta = np.deg2rad(param['beta'])

        # load initial state and mean
        init_state_mean = np.array(param['initial_state_mean'])
        init_state_cov = np.diag(param['initial_state_variance'])**2

        self.system_ = system_initialization(alphas, beta)

        self.filter_name = param['filter_name']
        self.Lie2Cart = param['Lie2Cart']

        # load world and landmarks
        if world is not None:
            self.world = world
            self.landmarks = self.world.getLandmarksInWorld()
        else:
            print("Plase provide a world with landmarks!")

        if self.filter_name is not None:
            print("Initializing", self.filter_name)
            self.filter_ = filter_initialization(self.system_, init_state_mean, init_state_cov, self.filter_name)
            self.state_ = self.filter_.getState()
        else:
            print("Please specify a filter name!")
        
        
        # load data.
        # in real-world application this should be a subscriber that subscribes to sensor topics
        # but for this homework example we load all data at once for simplicity
        self.data_handler = DataHandler()
        self.data = self.data_handler.load_2d_data()

        self.num_step = np.shape(self.data['motionCommand'])[0]

        self.pub = path_publisher()     # filter pose
        self.cmd_pub = path_publisher() # theoratical command path
        self.gt_pub = path_publisher()  # actual robot path
        self.landmark_visualizer = marker_publisher(self.world)

        self.loop_sleep_time = param['loop_sleep_time']

    def run_filter(self):
        
        results = np.zeros((self.num_step,7))
        for t in range(self.num_step):
            
            # get data for current timestamp
            motion_command = self.data['motionCommand'][t,:]
            observation = self.data['observation'][t,:]
            Y = self.data['Y'][t,:]
            Y2 = self.data['Y2'][t,:]

            if self.filter_name in ['EKF', 'UKF']:
                self.filter_.prediction(motion_command)
                self.filter_.correction(observation, self.landmarks)
            elif self.filter_name == 'PF':
                self.filter_.prediction(motion_command)
                self.filter_.correction(observation, self.landmarks)
            elif self.filter_name == "InEKF":
                self.filter_.prediction(motion_command)
                self.filter_.correction(Y, Y2, observation, self.landmarks)
            elif self.filter_name == "test":
                self.filter_.prediction(motion_command)
                self.filter_.correction(observation, self.landmarks)

            
            self.state_ = self.filter_.getState()
            
            # publisher 
            self.pub.publish_pose(self.state_)
            # self.pub.publish_path(self.state_)

            
            self.gt_pub.publish_gt_path(self.data['actual_state'][t])
            self.cmd_pub.publish_command_path(self.data['noise_free_state'][t])

            # visualize landmarks
            self.landmark_visualizer.publish_landmarks([observation[2], observation[5]])
            
            ## for plotting only
            if(self.Lie2Cart):
                results[t,:] = mahalanobis(self.state_,self.data['actual_state'][t],self.filter_name,self.Lie2Cart)

            rospy.sleep(self.loop_sleep_time)

        if(self.Lie2Cart):
            plot_error(results, self.data['actual_state'])
        
        

def main():

    rob_sys = RobotSystem()

    pass

if __name__ == '__main__':
    main()