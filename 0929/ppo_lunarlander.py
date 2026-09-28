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


def mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, out_dim),
    )


class ActorCritic(nn.Module):    
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.actor = mlp(state_dim, n_actions)
        self.critic = mlp(state_dim, 1)

    def forward(self, x):
        dist = Categorical(logits=self.actor(x))
        value = self.critic(x).squeeze(-1)
        return dist, value


def compute_gae(rewards, values, next_values, terminateds, episode_ends, gamma, gae_lambda):    
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0

    # TODO: Implement the computation of advantages using Generalized Advantage Estimation (GAE)

        
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


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
    rollout_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=3e-4,
    train_epochs=10,
    minibatch_size=64,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    checkpoint_path=None,
    weights_dir="weights",
    curve_path="ppo_lunarlander_curve.png",
):
    
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ActorCritic(state_dim, n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    episode_rewards = []
    episode_end_steps = []
    current_episode_reward = 0.0
    state, _ = env.reset()
    timesteps_done = 0

    while timesteps_done < total_timesteps:
        states, actions, log_probs, rewards, values = [], [], [], [], []
        terminateds, episode_ends = [], []
        truncation_bootstrap = {}

        for _ in range(rollout_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist, value = net(state_t)
                action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            values.append(value.item())
            terminateds.append(float(terminated))
            episode_ends.append(float(terminated or truncated))

            if truncated and not terminated:                
                with torch.no_grad():
                    _, v = net(torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0))
                truncation_bootstrap[len(rewards) - 1] = v.item()

            current_episode_reward += reward
            state = next_state
            timesteps_done += 1

            if terminated or truncated:
                episode_rewards.append(current_episode_reward)
                episode_end_steps.append(timesteps_done)
                current_episode_reward = 0.0
                state, _ = env.reset()

        with torch.no_grad():
            _, last_value = net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        next_values = values[1:] + [last_value.item()]
        for t, v in truncation_bootstrap.items():
            next_values[t] = v

        advantages, returns = compute_gae(rewards, values, next_values, terminateds, episode_ends,
                                          gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        indices = np.arange(rollout_steps)
        for _ in range(train_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, minibatch_size):
                idx = indices[start:start + minibatch_size]

                dist, values_pred = net(states_t[idx])
                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                # TODO: Implement the PPO loss                
                

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Timesteps {timesteps_done} | Episodes {len(episode_rewards)} | AvgReward(20) {avg_reward:.1f}")

    env.close()

    if checkpoint_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(weights_dir, f"ppo_lunarlander_{timestamp}.pt")
    torch.save(net.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

    plot_curve(episode_end_steps, episode_rewards, curve_path)
    return net


if __name__ == "__main__":
    print(f"Using device: {device}")
    train()
