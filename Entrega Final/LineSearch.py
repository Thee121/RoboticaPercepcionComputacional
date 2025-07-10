import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class LineSearch:
    def __init__(self, robot):
        self.robot = robot
        self.state = 0  # 0 = moving forward, 1 = turning
        self.step_counter = 0
        self.side_length = 5
        self.max_side = 50
        self.increment = 5
        self.repetitions = 0  # count how many times current side length has been used
        self.turn_direction = -5.0  # left turn
        self.moving = True

    def step_search(self):
        if self.is_line_found():
            self.robot.move(0.0, 0.0)
            print("Line found!")
            self.reset()
            return

        if self.state == 0:  # moving forward
            self.robot.move(1.0, 0.0)
            self.step_counter += 1
            if self.step_counter >= self.side_length:
                self.state = 1
                self.step_counter = 0
        elif self.state == 1:  # turning
            self.robot.move(0.0, self.turn_direction)
            self.state = 0
            self.repetitions += 1
            if self.repetitions >= 5:
                self.repetitions = 0
                self.side_length = min(self.side_length + self.increment, self.max_side)

    def is_line_found(self):
        image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", image)
        cv2.waitKey(1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, _ = findLineDeviation(gray)
        return found

    def reset(self):
        self.state = 0
        self.step_counter = 0
        self.side_length = 5
        self.repetitions = 0

