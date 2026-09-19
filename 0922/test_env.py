import random

import cv2

from snake_env import ACTION_MEANINGS, SnakeEnv

env = SnakeEnv(width=20, height=20)
obs = env.reset()

total_reward = 0.0
for step in range(500):
    env.render()

    action = random.randrange(env.n_actions)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if reward != 0:
        print(f"Step {step}: action={ACTION_MEANINGS[action]} reward={reward} score={info['score']}")

    if terminated or truncated:
        print(f"Episode end | terminated={terminated} truncated={truncated} | score={info['score']}")
        obs = env.reset()

    key = cv2.waitKey(150)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("驗證完成，隨機策略總 reward:", total_reward)
