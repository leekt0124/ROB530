import sys
sys.path.append('.')
import numpy as np

from utils.Landmark import *

class world2d:

    def __init__(self):
        

        self.num_landmarks_ = 6

        inner_offset = np.array([32,13])
        inner_size = np.array([420,270])
        complete_size = inner_size + 2*inner_offset

        marker_offset = np.array([21,0])
        marker_dist = np.array([442,292])

        marker_pos = np.zeros((6,2))

        # set marker position x
        marker_pos[0,0] = marker_offset[0]
        marker_pos[1,0] = marker_offset[0] + 0.5 * marker_dist[0]
        marker_pos[2,0] = marker_offset[0] + marker_dist[0]
        marker_pos[3,0] = marker_offset[0] + marker_dist[0]
        marker_pos[4,0] = marker_offset[0] + 0.5 * marker_dist[0]
        marker_pos[5,0] = marker_offset[0]

        # set marker position y
        marker_pos[0,1] = marker_offset[1]
        marker_pos[1,1] = marker_offset[1]
        marker_pos[2,1] = marker_offset[1]
        marker_pos[3,1] = marker_offset[1] + marker_dist[1]
        marker_pos[4,1] = marker_offset[1] + marker_dist[1]
        marker_pos[5,1] = marker_offset[1] + marker_dist[1]

        self.landmark_list_ = LandmarkList()
        self.marker_pos = marker_pos
        for i in range(self.num_landmarks_):
            
            lm = Landmark(i+1, marker_pos[i,:])

            self.landmark_list_.addLandmark(lm)

    def getLandmarksInWorld(self):
        return self.landmark_list_

    def getLandmark(self, id):
        return self.landmark_list_.getLandmark(id)

    def getNumLandmarks(self):
        return self.num_landmarks_

def main():

    robot_world = world2d()

    pass

if __name__ == '__main__':
    main()