import gymnasium as gym
import numpy as np


# =====================================================================================
# Customized your own reward function
# =====================================================================================
def custom_reward(obs, terminated, truncated, info):
    (
        x,                  # horizontal position, 0 = pad center, range [-2.5, 2.5]
        y,                  # height above ground, range [-2.5, 2.5]
        vx,                 # horizontal velocity, range [-10, 10]
        vy,                 # vertical velocity, range [-10, 10]
        angle,              # tilt angle in radians, 0 = level, range [-6.28, 6.28]
        angular_velocity,   # angular velocity, range [-10, 10]
        left_leg,           # left leg ground contact, 0 or 1
        right_leg,          # right leg ground contact, 0 or 1
    ) = obs

    reward = 0.0
    
    return reward


class LunarLanderRewardShaping(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = reward

        lander = self.env.unwrapped
        info["crashed"] = bool(lander.game_over) or abs(float(obs[0])) >= 1.0
        info["landed"] = terminated and not info["crashed"]

        reward = custom_reward(obs, terminated, truncated, info)
        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env = LunarLanderRewardShaping(env)
    return env
