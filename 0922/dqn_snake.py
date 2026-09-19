import os
import random
from collections import deque, namedtuple
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_env import SnakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class QNetwork(nn.Module):
    def __init__(self, width, height, n_actions, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * width * height, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, n_actions, epsilon):
    # TODO: Please complete the missing code for the epsilon-greedy action selection.
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return policy_net(state_t).argmax(dim=1).item()


def optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma):
    if len(buffer) < batch_size:
        return None

    batch = Transition(*zip(*buffer.sample(batch_size)))

    states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    # TODO: Please complete the missing code for the target Q-value computation and loss calculation.
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def moving_average(values, window=50):
    if len(values) < window:
        return np.array(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_curve(scores, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(scores, alpha=0.3, label="score")
    ma_score = moving_average(scores)
    ax.plot(range(len(scores) - len(ma_score), len(scores)), ma_score, label="moving avg (50)")
    ax.set_xlabel("episode")
    ax.set_ylabel("score (food eaten)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved training curve to {save_path}")


def train(
    num_episodes=2000,
    buffer_capacity=50_000,
    batch_size=64,
    gamma=0.9,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=50_000,
    target_update_freq=500,
    learning_starts=1_000,
    env_kwargs=None,
    checkpoint_path=None,
    weights_dir="weights",
    curve_path="dqn_snake_cnn_curve.png",
    render=False,
    render_every=50,
    render_delay_ms=30,
):
    env = SnakeEnv(**(env_kwargs or {}))
    n_actions = env.n_actions

    policy_net = QNetwork(env.width, env.height, n_actions).to(device)
    target_net = QNetwork(env.width, env.height, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    scores = []
    step_count = 0

    for episode in range(num_episodes):
        env.reset()
        state = env.get_image_obs()
        terminated = truncated = False
        info = {"score": 0}
        show_this_episode = render and episode % render_every == 0

        while not (terminated or truncated):
            if show_this_episode:
                env.render()
                cv2.waitKey(render_delay_ms)

            epsilon = max(epsilon_end, epsilon_start - step_count / epsilon_decay_steps)
            action = select_action(state, policy_net, n_actions, epsilon)

            _, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_image_obs()

            buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            step_count += 1

            if step_count > learning_starts:
                optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma)

            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        scores.append(info["score"])

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode} | Score {info['score']} | AvgScore(100) {avg_score:.2f} "
                  f"| Epsilon {epsilon:.3f} | Steps {step_count}")

    if render:
        cv2.destroyAllWindows()

    if checkpoint_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(weights_dir, f"dqn_snake_cnn_{timestamp}.pt")
    torch.save(policy_net.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

    plot_curve(scores, curve_path)
    return policy_net


if __name__ == "__main__":
    print(f"Using device: {device}")
    train()
