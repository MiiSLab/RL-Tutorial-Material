import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# 建立環境
env = gym.make("ALE/Pong-v5", render_mode="human")

obs, info = env.reset()
print("環境建立成功！")
print("可用動作數量:", env.action_space.n)
print("動作含義:", env.unwrapped.get_action_meanings())

total_reward = 0

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if reward != 0:
        print(f"Step {step}: 獲得 Reward = {reward}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("驗證完成，隨機策略總得分:", total_reward)