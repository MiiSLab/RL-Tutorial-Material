import argparse

import cv2
import numpy as np
import torch

from ppo_racecar import ActorCritic, RacecarEnv

FPS = 25                 # one frame per agent step (0.04 s of sim time) -> real-time playback
HOLD_LAST_FRAME_SEC = 2  # keep the final result on screen for a moment


def draw_text(frame, lines):
    for i, text in enumerate(lines):
        position = (10, 30 + i * 28)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def make_frame(env, student_id, info, status):
    frame = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
    progress = (info["lap"] - 1) + info["progress"]
    draw_text(frame, [
        f"Student: {student_id}",
        f"Time: {info['time']:.2f} s",
        f"Progress: {progress * 100:.1f} %",
        f"Status: {status}",
    ])
    return frame


def record(checkpoint_path, student_id, output_path=None):
    env = RacecarEnv(render_mode="rgb_array_birds_eye")
    net = ActorCritic(env.state_dim, env.action_dim)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()

    output_path = output_path or f"{student_id}.mp4"
    state, info = env.reset()
    frame = make_frame(env, student_id, info, "Running")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    writer.write(frame)

    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            dist, _ = net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        action = dist.mean.squeeze(0).numpy()  # use the mean: no random sampling
        state, _, terminated, truncated, info = env.step(action)

        if info["wall_collision"]:
            status = "Crashed"
        elif terminated:
            status = "Lap finished"
        elif truncated:
            status = "Time limit reached"
        else:
            status = "Running"
        frame = make_frame(env, student_id, info, status)
        writer.write(frame)

    for _ in range(HOLD_LAST_FRAME_SEC * FPS):
        writer.write(frame)
    writer.release()
    env.close()

    print(f"Result: {status} | sim time {info['time']:.2f} s | "
          f"progress {((info['lap'] - 1) + info['progress']) * 100:.1f} %")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--output", default=None, help="defaults to <student-id>.mp4")
    args = parser.parse_args()

    record(args.checkpoint, args.student_id, args.output)
