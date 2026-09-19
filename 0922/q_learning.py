import os
import pickle
import random
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

from snake_env import SnakeEnv


def make_q_table():
    return defaultdict(lambda: np.zeros(SnakeEnv.n_actions))


def select_action(q_table, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[state]))


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
    num_episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_episodes=3000,
    env_kwargs=None,
    q_table_path=None,
    weights_dir="weights",
    curve_path="qlearning_curve.png",
    render=False,
    render_every=50,
    render_delay_ms=30,
):
    env = SnakeEnv(**(env_kwargs or {}))
    q_table = make_q_table()

    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay_episodes)
        terminated = truncated = False
        info = {"score": 0}
        show_this_episode = render and episode % render_every == 0

        while not (terminated or truncated):
            if show_this_episode:
                env.render()
                cv2.waitKey(render_delay_ms)

            action = select_action(q_table, state, epsilon, env.n_actions)
            next_state, reward, terminated, truncated, info = env.step(action)

            best_next_value = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_value * (0.0 if terminated else 1.0)
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state

        scores.append(info["score"])

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode} | Score {info['score']} | AvgScore(100) {avg_score:.2f} "
                  f"| Epsilon {epsilon:.3f} | Q-table size {len(q_table)}")

    if render:
        cv2.destroyAllWindows()

    if q_table_path is None:
        os.makedirs(weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_table_path = os.path.join(weights_dir, f"snake_qtable_{timestamp}.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(dict(q_table), f)
    print(f"Saved Q-table ({len(q_table)} states) to {q_table_path}")

    plot_curve(scores, curve_path)
    return q_table


if __name__ == "__main__":
    train()
