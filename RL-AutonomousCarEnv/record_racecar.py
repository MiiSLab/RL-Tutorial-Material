import argparse
import os

import cv2
import numpy as np
import torch

from ppo_racecar import ActorCritic, RacecarEnv

FPS = 25                 # one frame per agent step (0.04 s of sim time) -> real-time playback
HOLD_LAST_FRAME_SEC = 2  # keep the final result on screen for a moment
WINDOW_NAME = "Racecar recording"


def draw_text(frame, lines):
    for i, text in enumerate(lines):
        position = (10, 30 + i * 28)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def make_frame(env, info, status):
    frame = cv2.cvtColor(np.ascontiguousarray(env.render()), cv2.COLOR_RGB2BGR)
    progress = (info["lap"] - 1) + info["progress"]
    draw_text(frame, [
        f"Time: {info['time']:.2f} s",
        f"Progress: {progress * 100:.1f} %",
        f"Status: {status}",
    ])
    return frame


def write_frame(writer, frame, preview):
    writer.write(frame)
    if preview:
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)  # lets the window redraw


def record(checkpoint_path, output_path=None, preview=True):
    env = RacecarEnv(render_mode="rgb_array_birds_eye")
    net = ActorCritic(env.state_dim, env.action_dim)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()

    output_path = output_path or os.path.splitext(os.path.basename(checkpoint_path))[0] + ".mp4"
    state, info = env.reset()
    frame = make_frame(env, info, "Running")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    write_frame(writer, frame, preview)

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
        frame = make_frame(env, info, status)
        write_frame(writer, frame, preview)

    for _ in range(HOLD_LAST_FRAME_SEC * FPS):
        writer.write(frame)
    writer.release()
    env.close()
    if preview:
        cv2.waitKey(HOLD_LAST_FRAME_SEC * 1000)
        cv2.destroyAllWindows()

    print(f"Result: {status} | sim time {info['time']:.2f} s | "
          f"progress {((info['lap'] - 1) + info['progress']) * 100:.1f} %")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="defaults to <checkpoint name>.mp4")
    parser.add_argument("--no-preview", action="store_true", help="record without showing a window")
    args = parser.parse_args()

    record(args.checkpoint, args.output, preview=not args.no_preview)
