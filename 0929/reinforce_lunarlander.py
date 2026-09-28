import os
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cpu")
torch.set_num_threads(1)

# False = Version A (vanilla: R(tau) for every step)
# True  = Version B (reward-to-go + baseline)
USE_REWARD_TO_GO_AND_BASELINE = True


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)


def select_action(policy, state):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # TODO: Implement the action selection using the policy network
    dist = policy(state_t)
    action = dist.sample()
    return action.item(), dist.log_prob(action).squeeze(0)


def compute_weights(rewards, gamma):
    if not USE_REWARD_TO_GO_AND_BASELINE:        
        # Sum of rewards (vanilla REINFORCE)
        # TODO: Implement the computation of weights for the vanilla REINFORCE algorithm
        return np.full(len(rewards), sum(rewards), dtype=np.float32)

    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0

    # Reward-to-go: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    # TODO: Implement the computation of returns using the rewards and discount factor gamma.    
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running

    # Baseline
    # TODO: Implement the computation of the baseline
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def moving_average(values, window=20):
    
    if len(values) < window:
        return np.array(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_curve(episode_end_steps, episode_rewards, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_end_steps, episode_rewards, alpha=0.3, label="episode reward")
    ma = moving_average(episode_rewards)
    ax.plot(episode_end_steps[len(episode_end_steps) - len(ma):], ma, label="moving avg (20)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved training curve to {save_path}")


def train(
    total_timesteps=300_000,
    gamma=0.99,
    lr=1e-3,
    checkpoint_path=None,
    weights_dir="weights",
    curve_path="reinforce_lunarlander_curve.png",
):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    episode_end_steps = []
    timesteps_done = 0
    episode = 0
    
    while timesteps_done < total_timesteps:
        state, _ = env.reset()
        log_probs = []
        rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            timesteps_done += 1

        # TODO: Implement the computation of loss
        weights = torch.as_tensor(compute_weights(rewards, gamma), dtype=torch.float32, device=device)
        loss = -(torch.stack(log_probs) * weights).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_end_steps.append(timesteps_done)

        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Timesteps {timesteps_done} | Episodes {episode + 1} | AvgReward(20) {avg_reward:.1f}")
        episode += 1

    env.close()

    if checkpoint_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(weights_dir, f"reinforce_lunarlander_{timestamp}.pt")
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"Saved policy to {checkpoint_path}")

    plot_curve(episode_end_steps, episode_rewards, curve_path)
    return policy


if __name__ == "__main__":
    print(f"Using device: {device}")
    train()
