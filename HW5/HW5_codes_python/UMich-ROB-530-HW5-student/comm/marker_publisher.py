import yaml

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from world.world2d import *
from utils.Landmark import *

class marker_publisher:
    def __init__(self, world):

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)
        
        marker_topic = param['landmark_topic']
        
        self.frame_id = param['marker_frame_id']
        self.world_dim = param['world_dimension']

        self.world = world   

        self.pub = rospy.Publisher(marker_topic,MarkerArray,queue_size=10)
    

    def publish_landmarks(self,observed_landmarks_id):

        markerArray = MarkerArray()
        for i in range(self.world.getNumLandmarks()):    

            lm = self.world.getLandmark(i+1)
            lm_pos = lm.getPosition()

            if not (lm.getID() in observed_landmarks_id):   # if the marker is not observed, use blue
                marker = Marker()
                marker.id = lm.getID()
                marker.header.frame_id = self.frame_id
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.scale.x = 10
                marker.scale.y = 10
                marker.scale.z = 10
                marker.color.a = 1.0
                # marker.color.r = 255.0/255.0
                # marker.color.g = 25/255.0
                # marker.color.b = 0/255.0
                marker.color.r = 0.0
                marker.color.g = 39/255.0
                marker.color.b = 76/255.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = lm_pos[0]
                marker.pose.position.y = lm_pos[1]
                marker.pose.position.z = 0

                markerArray.markers.append(marker)
            else:   # if the marker is observed, mark as maize
                marker = Marker()
                marker.id = lm.getID()
                marker.header.frame_id = self.frame_id
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.scale.x = 10
                marker.scale.y = 10
                marker.scale.z = 10
                marker.color.a = 1.0
                marker.color.r = 255.0/255.0
                marker.color.g = 203/255.0
                marker.color.b = 5/255.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = lm_pos[0]
                marker.pose.position.y = lm_pos[1]
                marker.pose.position.z = 0

                markerArray.markers.append(marker)
        
        self.pub.publish(markerArray)


