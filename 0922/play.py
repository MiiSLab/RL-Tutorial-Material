import argparse
import pickle

import cv2
import numpy as np
import torch

from snake_env import SnakeEnv


def load_q_table(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def select_action_qtable(q_table, state, n_actions):
    # Unseen states fall back to a zero vector -> argmax picks action 0 (STRAIGHT)
    q_values = q_table.get(state, np.zeros(n_actions))
    return int(np.argmax(q_values))


def select_action_dqn(policy_net, state, device):
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return policy_net(state_t).argmax(dim=1).item()


def play(checkpoint_path, episodes=10, env_kwargs=None, render=True, render_delay_ms=100):
    env = SnakeEnv(**(env_kwargs or {}))
    is_dqn = checkpoint_path.endswith(".pt")

    if is_dqn:
        from dqn_snake import QNetwork, device
        policy_net = QNetwork(env.width, env.height, env.n_actions).to(device)
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        policy_net.eval()
    else:
        q_table = load_q_table(checkpoint_path)

    scores = []
    for episode in range(episodes):
        env.reset()
        state = env.get_image_obs() if is_dqn else env.get_obs()
        terminated = truncated = False
        info = {"score": 0}

        while not (terminated or truncated):
            if render:
                env.render()
                cv2.waitKey(render_delay_ms)

            if is_dqn:
                action = select_action_dqn(policy_net, state, device)
            else:
                action = select_action_qtable(q_table, state, env.n_actions)

            _, reward, terminated, truncated, info = env.step(action)
            state = env.get_image_obs() if is_dqn else env.get_obs()

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
