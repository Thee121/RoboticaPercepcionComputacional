from pyrobot.brain import Brain
import cv2
import math
from pyrobot.tools.followLineTools import findLineDeviation

class FollowLine:
    def __init__(self, robot):
        self.robot = robot
        self.integral_area = 0.0
        self.previous_error = 0.0
        self.perpendicular_threshold = 80
        self.pid_params = {
            'Kp': 2.5,  # Ganancia proporcional
            'Ki': 0.03, # Ganancia integral
            'Kd': 0.2   # Ganancia derivativa
        }


    def get_image(self):
        cv_image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", cv_image)
        cv2.waitKey(1)
        
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        found_line, error_line = findLineDeviation(image_gray)

        self.error_line = error_line
        return found_line

    def follow_line(self):
        print("Siguiendo la línea...")
        self.integral_area += self.error_line

        if abs(self.error_line) > self.perpendicular_threshold:
            print("¡Línea casi perpendicular! Haciendo un giro más fuerte.")
            self.make_aggressive_turn()
            return

        # Calcular corrección usando PID
        correction_line = self.calculate_correction(self.error_line)
        # Aplicar límites a la corrección
        correction_line = self.apply_correction_limits(correction_line)
        # Calcular giro y velocidad con base en la corrección
        turn, speed = self.calculate_turn_and_speed(correction_line)

        self.robot.move(speed, turn)

    def make_aggressive_turn(self):
        turn_direction = -1 if self.error_line > 0 else 1 
        self.robot.move(0.0, 2.0 * turn_direction)
        cv2.waitKey(700)
        self.robot.move(0.5, 0.0)

    def calculate_correction(self, error):
        """Calcula la corrección PID"""
        return (self.pid_params['Kp'] * error + 
                self.pid_params['Ki'] * self.integral_area + 
                self.pid_params['Kd'] * (self.previous_error - error))

    def apply_correction_limits(self, correction):
        if -0.1 < correction < 0.1:
            correction = 0
        return correction

    def calculate_turn_and_speed(self, correction):
        turn = min(max(-1.0, -1.5 * correction), 1.0)
        speed = max(min(1.0, 1.2 - abs(correction)), 0.2)
        return turn, speed

