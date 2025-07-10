from pyrobot.brain import Brain
import cv2
import math
from pyrobot.tools.followLineTools import findLineDeviation

from FollowLine import FollowLine
from Avoid import Avoid
from LineSearch import LineSearch

class BrainFollowLine(Brain):
    def __init__(self, name, engine):
        super().__init__(name, engine)
        self.avoid = Avoid(self.robot)
        self.followLine = FollowLine(self.robot)
        self.lineSearch = LineSearch(self.robot)

    def setup(self):
        pass
            
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def step(self):
        foundLine = self.followLine.get_image()
        if not self.avoid.modeWall:
            self.avoid.check_wall()
        if self.avoid.modeWall:
            self.avoid.avoid_obstacle(foundLine)
        elif foundLine:
            self.followLine.follow_line()
        else:
            self.lineSearch.search_for_line()

def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)

