import argparse
import pickle

import cv2
import numpy as np

from snake_env import SnakeEnv


def load_q_table(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def select_action(q_table, state, n_actions):
    # Unseen states fall back to a zero vector -> argmax picks action 0 (STRAIGHT)
    q_values = q_table.get(state, np.zeros(n_actions))
    return int(np.argmax(q_values))


def play(checkpoint_path, episodes=10, env_kwargs=None, render=True, render_delay_ms=100):
    q_table = load_q_table(checkpoint_path)
    env = SnakeEnv(**(env_kwargs or {}))

    scores = []
    for episode in range(episodes):
        state = env.reset()
        terminated = truncated = False
        info = {"score": 0}

        while not (terminated or truncated):
            if render:
                env.render()
                cv2.waitKey(render_delay_ms)

            action = select_action(q_table, state, env.n_actions)
            state, reward, terminated, truncated, info = env.step(action)

        scores.append(info["score"])
        print(f"Episode {episode} | Score {info['score']}")

    if render:
        cv2.destroyAllWindows()

    scores = np.array(scores)
    print()
    print(f"Mean score: {scores.mean():.2f} +/- {scores.std():.2f}")
    print(f"Max score:  {scores.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    play(checkpoint_path=args.checkpoint, episodes=args.episodes, render=not args.no_render)
