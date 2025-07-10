from pyrobot.brain import Brain
import cv2
import math
from pyrobot.tools.followLineTools import findLineDeviation

class LineSearch:
    def __init__(self, robot):
        self.robot = robot
        self.search_speed = 2.0
        self.base_side_length = 5.0
        self.side_length_increment = 5.0
        self.max_side_length = 50.0

    def search_for_line(self):
        print("Starting line search")
        side_length = self.base_side_length
        
        while not self.is_line_found():
            print(f"Searching with square side length {side_length}...")
            self.execute_square_pattern(side_length)
            if not self.is_line_found():
                side_length = min(side_length + self.side_length_increment, self.max_side_length)
        
        print("Line found!")
        self.robot.move(0.0, 0.0)

    def execute_square_pattern(self, side_length):
        for _ in range(4):
            if self.is_line_found():
                self.robot.move(0.0, 0.0)
                return
            
            self.move_forward(side_length)
            self.turn_90_degrees()
            
    def move_forward(self, distance):
        moved_distance = 0.0
        start_time = cv2.getTickCount()
        
        while moved_distance < distance:
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            moved_distance = self.search_speed * elapsed_time
            self.robot.move(1.0, 0.0)
            
            if self.is_line_found():
                self.robot.move(0.0, 0.0)
                return
        
        self.robot.move(0.0, 0.0)

    def turn_90_degrees(self):
        print("Turning to the right.")
        self.robot.move(0.0, -3) 
        cv2.waitKey(500)
        self.robot.move(0.0, 0.0)

    def is_line_found(self):
        cv_image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", cv_image)
        cv2.waitKey(1)
        
        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        foundLine, errorLine = findLineDeviation(imageGray)
        
        return foundLine

