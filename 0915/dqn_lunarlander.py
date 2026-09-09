import os
import random
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env import make_env
from model import QNetwork, device

print(f"Using device: {device}")
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


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
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
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

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    num_episodes=1000,
    buffer_capacity=100_000,
    batch_size=64,
    gamma=0.99,
    lr=5e-4,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=100_000,
    target_update_freq=1000,
    learning_starts=1_000,
    checkpoint_path=None,
    weights_dir="weights",
    log_dir="runs/dqn_lunarlander",
):
    if checkpoint_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(weights_dir, f"dqn_lunarlander_{timestamp}.pt")
    print(f"Checkpoints will be saved to: {checkpoint_path}")

    env = make_env()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    policy_net = QNetwork(state_dim, n_actions).to(device)
    target_net = QNetwork(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    writer = SummaryWriter(log_dir=log_dir)

    step_count = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            epsilon = max(epsilon_end, epsilon_start - step_count / epsilon_decay_steps)
            action = select_action(state, policy_net, n_actions, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step_count += 1

            if step_count > learning_starts:
                optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma)

            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar("reward", episode_reward, episode)
        writer.add_scalar("epsilon", epsilon, episode)

        print(f"Episode {episode} | Reward {episode_reward:.2f} | Epsilon {epsilon:.3f} | Steps {step_count}")

        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), checkpoint_path)

    env.close()
    torch.save(policy_net.state_dict(), checkpoint_path)
    writer.close()


if __name__ == "__main__":
    train()
