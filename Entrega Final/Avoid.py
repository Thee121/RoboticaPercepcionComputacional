import cv2
import numpy as np

class Avoid:
    def __init__(self, robot):
        self.robot = robot
        self.avoiding = False
        self.phase = None
        self.forward_steps = 0
        self.realign_steps = 0

    def step(self, found_line):
        if not self.avoiding:
            if self.obstacle_in_view():
                print("Obstacle detected.")
                self.avoiding = True
                self.phase = "turning_right"
        else:
            self._avoid_step(found_line)

    def _avoid_step(self, found_line):
        if self.phase == "turning_right":
            self.robot.move(0.0, -1.0)  # turn right slowly
            if not self.obstacle_in_view():
                self.phase = "initial_move_forward"

        elif self.phase == "initial_move_forward":
            self.robot.move(8.0, 0.0)
            self.forward_steps += 1
            
            if(self.forward_steps >= 3):
                self.phase = "check_left"

        elif self.phase == "check_left":
            self.robot.move(0.0, 1.0)  # test turn left
            if self.obstacle_in_view():
                self.robot.move(0.0, -1.0)  # undo left
            else:
                self.phase = "realign"

        elif self.phase == "realign":
            self.robot.move(8.0, 0.0)
            self.realign_steps += 1
            
            if(self.realign_steps >= 2):
                self.phase = "realign"
                
            self.robot.move(0.0, 1.0)     # gentle left turn
            if found_line:
                self.reset()
                return
            else:
                self_realign_steps = 0
    def reset(self):
        self.avoiding = False
        self.phase = None
        self.forward_steps = 0
        self.realign_steps = 0

    def obstacle_in_view(self):
        frame = self.robot.getImage()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 50, 50])
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.countNonZero(mask) > 50
