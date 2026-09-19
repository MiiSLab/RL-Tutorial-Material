import random
from collections import deque

import cv2
import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_DELTA = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}
CLOCKWISE = [UP, RIGHT, DOWN, LEFT]

STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
ACTION_MEANINGS = ["STRAIGHT", "TURN_RIGHT", "TURN_LEFT"]


class SnakeEnv:
    """Simple grid Snake environment for tabular Q-learning.

    Observation is an 11-dim binary tuple:
        (danger_straight, danger_right, danger_left,
         moving_up, moving_down, moving_left, moving_right,
         food_up, food_down, food_left, food_right)

    Actions are relative to the current heading (not absolute directions):
        0 = STRAIGHT, 1 = TURN_RIGHT, 2 = TURN_LEFT
    """

    n_actions = 3
    n_obs_features = 11

    def __init__(self, width=10, height=10, max_steps_per_food=100):
        self.width = width
        self.height = height
        self.max_steps_per_food = max_steps_per_food
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.direction = RIGHT
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.score = 0
        self.steps_since_food = 0
        self.food = None
        self._place_food()
        return self.get_obs()

    def _place_food(self):
        while True:
            pos = (random.randrange(self.width), random.randrange(self.height))
            if pos not in self.snake:
                self.food = pos
                return

    def _out_of_bounds(self, pos):
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def _hits_body(self, pos, ignore_tail=False):
        body = list(self.snake)[:-1] if ignore_tail else list(self.snake)
        return pos in body

    def _is_danger(self, pos):
        # Conservative check for the observation's danger features: treats
        # the tail cell as unsafe even though it will vacate next step
        # (unless the snake is about to eat). Good enough as a learning hint.
        return self._out_of_bounds(pos) or self._hits_body(pos)

    def get_obs(self):
        head = self.snake[0]
        idx = CLOCKWISE.index(self.direction)
        straight_dir = self.direction
        right_dir = CLOCKWISE[(idx + 1) % 4]
        left_dir = CLOCKWISE[(idx - 1) % 4]

        def next_pos(direction):
            dx, dy = DIRECTION_DELTA[direction]
            return (head[0] + dx, head[1] + dy)

        danger_straight = self._is_danger(next_pos(straight_dir))
        danger_right = self._is_danger(next_pos(right_dir))
        danger_left = self._is_danger(next_pos(left_dir))

        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        return (
            int(danger_straight), int(danger_right), int(danger_left),
            int(self.direction == UP), int(self.direction == DOWN),
            int(self.direction == LEFT), int(self.direction == RIGHT),
            int(food_up), int(food_down), int(food_left), int(food_right),
        )

    def get_image_obs(self):
        # Channel-encoded grid instead of rendered pixels: 0=body, 1=head, 2=food.
        # A single frame is a full Markov state here (unlike Pong), since the
        # snake's own body already encodes its recent trajectory.
        img = np.zeros((3, self.height, self.width), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            img[1 if i == 0 else 0, y, x] = 1.0
        fx, fy = self.food
        img[2, fy, fx] = 1.0
        return img

    def step(self, action):
        idx = CLOCKWISE.index(self.direction)
        if action == STRAIGHT:
            self.direction = self.direction
        elif action == TURN_RIGHT:
            self.direction = CLOCKWISE[(idx + 1) % 4]
        elif action == TURN_LEFT:
            self.direction = CLOCKWISE[(idx - 1) % 4]
        else:
            raise ValueError(f"Invalid action: {action}")

        head = self.snake[0]
        dx, dy = DIRECTION_DELTA[self.direction]
        new_head = (head[0] + dx, head[1] + dy)
        ate_food = new_head == self.food

        terminated = self._out_of_bounds(new_head) or self._hits_body(new_head, ignore_tail=not ate_food)
        truncated = False
        reward = 0.0

        if terminated:
            reward = -10.0
        else:
            self.snake.appendleft(new_head)
            if ate_food:
                reward = 10.0
                self.score += 1
                self.steps_since_food = 0
                self._place_food()
            else:
                self.snake.pop()
                self.steps_since_food += 1

            if self.steps_since_food > self.max_steps_per_food * len(self.snake):
                truncated = True

        info = {"score": self.score}
        return self.get_obs(), reward, terminated, truncated, info

    def render(self, cell_size=30, window_name="Snake"):
        status_bar_height = 30
        img = np.zeros((self.height * cell_size + status_bar_height, self.width * cell_size, 3), dtype=np.uint8)

        for i, (x, y) in enumerate(self.snake):
            color = (0, 0, 255) if i == 0 else (100, 100, 100)  # BGR: red head, gray body
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size - 1, (y + 1) * cell_size - 1)
            cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

        fx, fy = self.food
        cv2.rectangle(
            img,
            (fx * cell_size, fy * cell_size),
            ((fx + 1) * cell_size - 1, (fy + 1) * cell_size - 1),
            (0, 255, 0),  # green
            thickness=-1,
        )

        cv2.putText(img, f"Score: {self.score}", (5, self.height * cell_size + status_bar_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
