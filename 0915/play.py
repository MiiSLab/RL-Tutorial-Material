import argparse

import numpy as np
import torch

from env import make_env
from model import QNetwork, device


def run_episode(env, policy_net):
    state, _ = env.reset()
    episode_reward = 0.0
    last_raw_reward = 0.0
    done = False

    while not done:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_t).argmax(dim=1).item()

        state, reward, terminated, truncated, info = env.step(action)        
        done = terminated or truncated
        last_raw_reward = info.get("raw_reward", reward)
        episode_reward += last_raw_reward

    landed = last_raw_reward >= 50
    crashed = last_raw_reward <= -50
    return episode_reward, landed, crashed


def play(checkpoint_path="dqn_lunarlander.pt", episodes=5, render=True):
    env = make_env(render_mode="human" if render else None)
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    policy_net = QNetwork(state_dim, n_actions).to(device)
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy_net.eval()

    rewards = []
    landed_count = 0
    crashed_count = 0

    for episode in range(episodes):
        episode_reward, landed, crashed = run_episode(env, policy_net)
        rewards.append(episode_reward)
        landed_count += landed
        crashed_count += crashed

        outcome = "Landed" if landed else "Crashed" if crashed else "Timeout"
        print(f"Episode {episode} | Reward {episode_reward:.2f} | {outcome}")

    env.close()

    rewards = np.array(rewards)
    print()
    print(f"Mean reward:          {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"Landing success rate: {landed_count / episodes * 100:.1f}% ({landed_count}/{episodes})")
    print(f"Crash rate:           {crashed_count / episodes * 100:.1f}% ({crashed_count}/{episodes})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dqn_lunarlander.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    play(checkpoint_path=args.checkpoint, episodes=args.episodes, render=not args.no_render)
