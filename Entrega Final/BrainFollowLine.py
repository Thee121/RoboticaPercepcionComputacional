from pyrobot.brain import Brain
import cv2
from pyrobot.brain import Brain
from Avoid import Avoid
from FollowLine import FollowLine
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
        pass

    def step(self):
        found_line = self.followLine.get_image()
        self.avoid.step(found_line)

        if not self.avoid.avoiding:
            if found_line:
                self.followLine.follow_line()
            else:
                self.lineSearch.step_search()
        
def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
      engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)

