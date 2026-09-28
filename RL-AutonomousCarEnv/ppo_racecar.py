import os
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import racecar_gym.envs.gym_api  # registers the racecar environments

device = torch.device("cpu")
torch.set_num_threads(1)

SCENARIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "circle_cw.yml")
TIME_LIMIT = 120.0          # must match time_limit in the scenario file
LIDAR_STRIDE = 10           # keep every 10th ray: 1080 -> 108
LIDAR_MAX_RANGE = 15.0
VELOCITY_SCALE = 5.0        # the motor's max velocity
ACTION_REPEAT = 4           # physics runs at 100 Hz but the lidar only updates at 25 Hz
PROGRESS_SCALE = 100.0      # one full lap = +100
COLLISION_PENALTY = -10.0


class RacecarEnv(gym.Wrapper):
    # Turns the dict observation into a flat vector, takes a 2-D action array
    # [motor, steering], and computes our own reward and termination.

    state_dim = 1080 // LIDAR_STRIDE + 6
    action_dim = 2

    def __init__(self, render_mode="rgb_array_birds_eye"):
        super().__init__(gym.make("SingleAgentRaceEnv-v0", scenario=SCENARIO, render_mode=render_mode))
        self._last_progress = 0.0

    def _process_obs(self, obs):
        lidar = obs["lidar"][::LIDAR_STRIDE] / LIDAR_MAX_RANGE
        velocity = obs["velocity"] / VELOCITY_SCALE
        return np.concatenate([lidar, velocity]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(options=dict(mode="grid"))
        self._last_progress = info["lap"] + info["progress"]
        return self._process_obs(obs), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        env_action = {"motor": action[:1], "steering": action[1:]}

        reward = 0.0
        terminated = truncated = False
        for _ in range(ACTION_REPEAT):
            obs, _, done, _, info = self.env.step(env_action)

            # Signed progress: driving backwards gives a negative reward.
            progress = info["lap"] + info["progress"]
            delta = progress - self._last_progress
            delta = (delta + 0.5) % 1.0 - 0.5  # guard against jumps at the start line
            self._last_progress = progress
            reward += PROGRESS_SCALE * delta

            if info["wall_collision"]:
                reward += COLLISION_PENALTY
                terminated = True
            elif done:
                # The env reports both "lap finished" and "time is up" as done;
                # only the time limit is a truncation.
                truncated = info["time"] > TIME_LIMIT
                terminated = not truncated

            if terminated or truncated:
                break

        return self._process_obs(obs), reward, terminated, truncated, info

    def close(self):
        super().close()
        # racecar_gym never disconnects PyBullet and sends every call to client 0,
        # so without this a second env would load a duplicate track and car into the same world.
        if pybullet.isConnected():
            pybullet.disconnect()


def mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = mlp(state_dim, action_dim)
        # State-independent std; log_std = -0.5 means an initial std of about 0.6
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = mlp(state_dim, 1)

    def forward(self, x):
        # TODO: Implement the forward pass to return a Normal distribution
        mean = self.actor_mean(x)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        value = self.critic(x).squeeze(-1)
        return dist, value


def compute_gae(rewards, values, next_values, terminateds, episode_ends, gamma, gae_lambda):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminateds[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - episode_ends[t]) * gae
        advantages[t] = gae
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
    ax.set_xlabel("agent steps")
    ax.set_ylabel("episode reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved training curve to {save_path}")


def train(
    total_timesteps=50_000,
    rollout_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=3e-4,
    train_epochs=10,
    minibatch_size=64,
    value_coef=0.5,
    entropy_coef=0.0,       # exploration comes from log_std in continuous control
    max_grad_norm=0.5,
    checkpoint_path=None,
    weights_dir="weights",
    curve_path="ppo_racecar_curve.png",
):
    env = RacecarEnv()
    net = ActorCritic(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    if checkpoint_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(weights_dir, f"ppo_racecar_{timestamp}.pt")

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

            # The env clips to [-1, 1]; store the raw sample so log_prob stays consistent.
            raw_action = action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(raw_action)

            states.append(state)
            actions.append(raw_action)
            log_probs.append(dist.log_prob(action).sum(-1).item())
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
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.float32, device=device)
        old_log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        indices = np.arange(rollout_steps)
        for _ in range(train_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, minibatch_size):
                idx = indices[start:start + minibatch_size]

                dist, values_pred = net(states_t[idx])

                # TODO: Implement the PPO loss
                new_log_probs = dist.log_prob(actions_t[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values_pred, returns_t[idx])

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        # Training takes a while, so keep the latest weights on disk after every update.
        torch.save(net.state_dict(), checkpoint_path)

        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Timesteps {timesteps_done} | Episodes {len(episode_rewards)} | "
                  f"AvgReward(20) {avg_reward:.1f} | std {net.log_std.exp().detach().cpu().numpy().round(2)}")

    env.close()
    print(f"Saved model to {checkpoint_path}")
    plot_curve(episode_end_steps, episode_rewards, curve_path)
    return net


if __name__ == "__main__":
    print(f"Using device: {device}")
    train()
