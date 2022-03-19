import rospy
import numpy as np
from copy import deepcopy, copy

class Landmark:
    '''
    Landmark holds the ID and a true position of a landmark object.
    '''
    def __init__(self, id=None, position=None):
        self.id_ = None
        self.dim_ = None
        self.position_ = None

        if id is not None and isinstance(id,int):
            self.id_ = id
        else:
            self.id_ = -1
            print("landmark id is not provided!")

        if position is not None and isinstance(position, np.ndarray):
            self.position_ = copy(position)
        elif position is not None and isinstance(position, list):
            self.position_ = np.array(position)
        else:
            print("landmark position is not initialized!")

    def setID(self, id):
        if id is not None and isinstance(id,int):
            self.id_ = id

    def getID(self):
        return self.id_

    def setPosition(self, position):
        if position is not None and isinstance(position, np.ndarray):
            self.position_ = np.copy(position)
        elif position is not None and isinstance(position, list):
            self.position_ = np.array(position)
        else:
            print("landmark position is not set!")

    def getPosition(self):
        return self.position_

class LandmarkList:    
    
    def __init__(self):
        self.landmarks_ = {}   
        self.num_landmarks_ = 0

    def addLandmark(self, landmark):
        id = landmark.getID()
        if id in self.landmarks_:
            print("duplicate landmark ID exist. Overriding...")
        else:
            self.num_landmarks_ = self.num_landmarks_ + 1

        self.landmarks_[landmark.getID()] = deepcopy(landmark)

    def getLandmark(self, id):
        return self.landmarks_[id]

    def printID(self):
        print("The landmark list contains landmark IDs:")
        for id in self.landmarks_:
            print(id)

    def getNumLandmarks(self):
        return self.num_landmarks_

def main():
    pos = 5
    lm1 = Landmark(id=1,position=[1,3])
    lm2 = Landmark(id=5,position=[5,6])
    lm_list = LandmarkList()
    lm_list.addLandmark(lm1)
    lm_list.addLandmark(lm2)
    lm3 = lm_list.getLandmark(1)
    print("ID: ", lm3.getID(), "positoin: ", lm3.getPosition())


if __name__ == '__main__':
    main()