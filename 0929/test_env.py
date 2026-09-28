import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset()
print("Successfully created environment!")
print("Number of available actions:", env.action_space.n)
print("Action meanings:", ["NOOP", "FIRE_LEFT_ENGINE", "FIRE_MAIN_ENGINE", "FIRE_RIGHT_ENGINE"])
print("Observation space:", env.observation_space)

total_reward = 0
episode_reward = 0

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_reward += reward

    if terminated or truncated:
        print(f"Step {step}: Episode finished with reward {episode_reward:.1f}")
        episode_reward = 0
        obs, info = env.reset()

env.close()
print(f"Total reward: {total_reward:.1f}")
