
import sys
from tkinter.ttk import Labelframe
sys.path.append('.')

import yaml

from scipy.spatial.transform import Rotation
from scipy.linalg import expm, logm

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from system.RobotState import *

class path_publisher:
    def __init__(self):

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)
        
        pose_topic = param['pose_topic']
        path_topic = param['path_topic']
        gt_path_topic = param['gt_path_topic']
        command_path_topic = param['command_path_topic']
        ellipse_topic = param['ellipse_topic']

        self.path_frame = param['path_frame_id']

        self.filter_name = param['filter_name']
        self.world_dim = param['world_dimension']

        self.path = Path()
        self.path.header.frame_id = self.path_frame

        self.gt_path = Path()
        self.gt_path.header.frame_id = self.path_frame

        self.pose_pub = rospy.Publisher(pose_topic,PoseWithCovarianceStamped,queue_size=100)
        self.path_pub = rospy.Publisher(path_topic,Path,queue_size=10)
        self.gt_path_pub = rospy.Publisher(gt_path_topic,Path,queue_size=10)
        self.cmd_path_pub = rospy.Publisher(command_path_topic,Path,queue_size=10)
        self.ellipse_pub = rospy.Publisher(ellipse_topic,Marker,queue_size=10)

    def publish_pose(self, state):

        position = state.getPosition()
        orientation = state.getOrientation()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = self.path_frame
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]

        if self.world_dim == 2:
            msg.pose.pose.position.z = 0
            rot = Rotation.from_euler('z',orientation,degrees=False)
            quat = rot.as_quat()    # (x, y, z, w)
            msg.pose.pose.orientation.x = quat[0]
            msg.pose.pose.orientation.y = quat[1]
            msg.pose.pose.orientation.z = quat[2]
            msg.pose.pose.orientation.w = quat[3]

            cov = np.zeros((6,6))
            if self.filter_name != "InEKF":
                cov[0:2,0:2] = state.getCovariance()[0:2,0:2]
            else:
                ellipse_line_msg = self.make_ellipse(state)
                self.ellipse_pub.publish(ellipse_line_msg)

            # We wish to visualize 3 sigma contour
            msg.pose.covariance = np.reshape(9*cov,(-1,)).tolist()
            

        self.pose_pub.publish(msg)
        

    def make_ellipse(self,state):
        G1 = np.array([[0,0,1],
                    [0,0,0],
                    [0,0,0]])
    
        G2 = np.array([[0,0,0],
                        [0,0,1],
                        [0,0,0]])
        
        G3 = np.array([[0,-1,0],
                        [1,0,0],
                        [0,0,0]])

        phi = np.arange(-np.pi, np.pi, 0.01)
        circle = np.array([np.cos(phi), np.sin(phi), np.zeros(np.size(phi))]).T

        scale = np.sqrt(7.815)

        mean = state.getState()
        mean_matrix = np.eye(3)
        R = np.array([[np.cos(mean[2]),-np.sin(mean[2])],[np.sin(mean[2]),np.cos(mean[2])]])

        mean_matrix[0:2,0:2] = R
        mean_matrix[0,2] = mean[0]
        mean_matrix[1,2] = mean[1]

        ellipse_line_msg = Marker()
        ellipse_line_msg.type = Marker.LINE_STRIP
        ellipse_line_msg.id = 99
        ellipse_line_msg.header.stamp = rospy.get_rostime()
        ellipse_line_msg.header.frame_id = self.path_frame
        ellipse_line_msg.action = Marker.ADD
        ellipse_line_msg.scale.x = 2
        ellipse_line_msg.scale.y = 2
        ellipse_line_msg.scale.z = 2
        ellipse_line_msg.color.a = 0.6
        ellipse_line_msg.color.r = 239/255.0
        ellipse_line_msg.color.g = 41/255.0
        ellipse_line_msg.color.b = 41/255.0
        
        L = np.linalg.cholesky(state.getCovariance())
        for j in range(np.shape(circle)[0]):
            ell_se2_vec = scale * L @ circle[j,:].reshape(-1,1)
            temp = expm(G1 * ell_se2_vec[0] + G2 * ell_se2_vec[1] + G3 * ell_se2_vec[2]) @ mean_matrix
            ellipse_point = Point()
            ellipse_point.x = temp[0,2]
            ellipse_point.y = temp[1,2]
            ellipse_point.z = 0
            ellipse_line_msg.points.append(ellipse_point)

        return ellipse_line_msg


    def publish_state_path(self, state):

        position = state.getPosition()
        orientation = state.getOrientation()

        pose = PoseStamped()
        pose.header.stamp = rospy.get_rostime()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]

        if self.world_dim == 2:
            pose.pose.position.z = 0
            rot = Rotation.from_euler('z',orientation,degrees=False)
            quat = rot.as_quat()    # (x, y, z, w)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

        self.path.poses.append(pose)

        self.path_pub.publish(self.path)

    def publish_gt_path(self, data):

        x = data[0]
        y = data[1]
        theta = data[2]

        pose = PoseStamped()
        pose.header.stamp = rospy.get_rostime()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = x
        pose.pose.position.y = y

        pose.pose.position.z = 0
        rot = Rotation.from_euler('z',theta,degrees=False)
        quat = rot.as_quat()    # (x, y, z, w)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        self.path.poses.append(pose)

        self.gt_path_pub.publish(self.path)

    def publish_command_path(self,data):

        x = data[0]
        y = data[1]
        theta = data[2]

        pose = PoseStamped()
        pose.header.stamp = rospy.get_rostime()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = x
        pose.pose.position.y = y

        pose.pose.position.z = 0
        rot = Rotation.from_euler('z',theta,degrees=False)
        quat = rot.as_quat()    # (x, y, z, w)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        self.path.poses.append(pose)

        self.cmd_path_pub.publish(self.path)

def main():

    state = RobotState()
    path_pub = path_publisher()

    path_pub.make_ellipse()

    pass

if __name__ == '__main__':
    main()