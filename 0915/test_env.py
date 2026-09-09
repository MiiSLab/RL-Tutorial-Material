import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")

obs, info = env.reset()
print("Successfully created environment!")
print("Number of available actions:", env.action_space.n)
print("Action meanings:", env.unwrapped.get_action_meanings())

total_reward = 0

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if reward != 0:
        print(f"Step {step}: Reward received: {reward}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print(f"Total reward: {total_reward}")