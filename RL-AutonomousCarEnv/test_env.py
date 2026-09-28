import os
import time

import gymnasium as gym
import numpy as np

import racecar_gym.envs.gym_api  # registers the racecar environments

RENDER_MODE = "human"  # opens the PyBullet window (Ctrl + mouse drag to move the camera)
SCENARIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "circle_cw.yml")

env = gym.make("SingleAgentRaceEnv-v0", scenario=SCENARIO, render_mode=RENDER_MODE)

obs, info = env.reset(options=dict(mode="grid"))
print("Successfully created environment!")
print("Action space:", env.action_space)
print("Observations:")
for name, value in obs.items():
    print(f"  {name}: shape {np.shape(value)}")

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if info["wall_collision"] or done:
        reason = "hit the wall" if info["wall_collision"] else "episode finished"
        print(f"Step {step}: {reason} | time {info['time']:.2f} s | "
              f"progress {(info['lap'] - 1 + info['progress']) * 100:.1f} %")
        obs, info = env.reset(options=dict(mode="grid"))

    if RENDER_MODE == "human":
        time.sleep(0.01)  # one physics step is 0.01 s, so this plays roughly in real time

env.close()
print("Environment check finished!")
