class Avoid:
    def __init__(self, robot):
        self.robot = robot
        self.modeWall = False
        self.state = 'IDLE'
        self.turn_steps = 0
        self.advance_steps = 0
        self.search_steps = 0

    def check_wall(self):
        front_distance = min(s.distance() for s in self.robot.range["front"])
        if front_distance <= 0.7:
            if not self.modeWall:
                print("Obstacle detected.")
            self.start_avoidance()

    def start_avoidance(self):
        self.modeWall = True
        self.state = 'TURN_LEFT'
        self.turn_steps = 0
        self.advance_steps = 0
        self.search_steps = 0

    def avoid_obstacle(self, found_line):
        if self.state == 'TURN_LEFT':
            self.turn_left()
        if self.state == 'FOLLOW_WALL':
            self.follow_wall()
        if self.state == 'ADVANCE_AFTER_OBSTACLE':
            self.advance_after_obstacle()
        if self.state == 'RETURN_TO_LINE':
            self.return_to_line(found_line)
        if self.state == 'ADVANCE_TURN_LOOP':
            self.advance_turn_loop(found_line)

    def turn_left(self):
        self.robot.move(0.0, 2)
        self.turn_steps += 1
        if self.turn_steps >= 8:
            self.state = 'FOLLOW_WALL'

    def follow_wall(self):
        self.robot.move(1, 0.0)
        self.advance_steps += 1
        if self.advance_steps >= 15:
            self.state = 'ADVANCE_AFTER_OBSTACLE'
            self.advance_steps = 0

    def advance_after_obstacle(self):
        self.robot.move(1.5, 0.0)
        self.advance_steps += 1
        if self.advance_steps >= 15:
            self.state = 'RETURN_TO_LINE'
            self.turn_steps = 0

    def return_to_line(self, found_line):
        if not found_line:
            self.robot.move(0.0, -1.5)
            self.turn_steps += 1
            if self.turn_steps >= 8:
                self.state = 'ADVANCE_TURN_LOOP'
                self.turn_steps = 0
                self.advance_steps = 0

        if found_line:
            print("Line found!")
            self.reset()

    def advance_turn_loop(self, found_line):
        if found_line:
            print("Line found!")
            self.reset()
            return
        
        if self.advance_steps < 15:
            self.robot.move(2, 0.0)
            self.advance_steps += 1
        else:
            self.robot.move(0.0, -2)
            self.turn_steps += 1
            if self.turn_steps >= 15:
                self.advance_steps = 0
                self.turn_steps = 0

    def reset(self):
        self.modeWall = False
        self.state = 'IDLE'
        self.turn_steps = 0
        self.advance_steps = 0
        self.search_steps = 0

