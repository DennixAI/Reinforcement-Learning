import argparse
import glob
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

try:
    from vizdoom import gymnasium_wrapper as _vizdoom_registry  # type: ignore  # noqa: F401
except ImportError:
    _vizdoom_registry = None


@dataclass
class HistoryBook:
    losses: List[Tuple[int, float]]
    epsilons: List[Tuple[int, float]]
    episodic: List[Tuple[int, float, int]]
    evaluations: List[Tuple[int, float]]


class DoomQNetwork(nn.Module):
    def __init__(self, channels: int, actions: int, height: int, width: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv_c = nn.Conv2d(32, 64, kernel_size=3, stride=3)

        with torch.no_grad():
            probe = torch.zeros(1, channels, height, width)
            probe = self.conv_c(self.conv_b(self.conv_a(probe)))
            flattened = probe.view(1, -1).shape[1]

        self.linear_a = nn.Linear(flattened, 512)
        self.linear_b = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, actions)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.relu(self.conv_a(data))
        data = torch.relu(self.conv_b(data))
        data = torch.relu(self.conv_c(data))
        data = torch.flatten(data, 1)
        data = torch.relu(self.linear_a(data))
        data = torch.relu(self.linear_b(data))
        return self.output_layer(data)


class DuelingDoomQNetwork(nn.Module):
    """Dueling DQN head sharing the same conv trunk.

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """
    def __init__(self, channels: int, actions: int, height: int, width: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv_c = nn.Conv2d(32, 64, kernel_size=3, stride=3)

        with torch.no_grad():
            probe = torch.zeros(1, channels, height, width)
            probe = self.conv_c(self.conv_b(self.conv_a(probe)))
            flattened = probe.view(1, -1).shape[1]

        self.fc = nn.Linear(flattened, 512)
        self.value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.adv = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_a(x))
        x = torch.relu(self.conv_b(x))
        x = torch.relu(self.conv_c(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        v = self.value(x)
        a = self.adv(x)
        return v + a - a.mean(dim=1, keepdim=True)


def _normalize_frame(frame: np.ndarray, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    frame = frame.astype(np.float32)
    tensor = torch.from_numpy(frame)

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    if tensor.shape[0] == 3:
        grayscale = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        tensor = grayscale.unsqueeze(0)

    tensor = tensor.unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    tensor = tensor.squeeze(0) / 255.0
    return tensor.to(device)

def prepare_input(obs, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    # handle dict obs like {"screen": ...}
    payload = obs.get("screen", obs) if isinstance(obs, dict) else obs
    arr = np.array(payload, copy=False).astype(np.float32)

    # 4D: stacked frames -> normalize each and stack
    if arr.ndim == 4:
        frames = []
        for f in arr:
            f = f.astype(np.float32)
            if f.ndim == 3 and f.shape[-1] in (1, 3):
                f = np.transpose(f, (2, 0, 1))
            if f.ndim == 2:
                f = f[None, ...]
            t = torch.from_numpy(f).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            if t.shape[0] == 3:  # rgb -> gray
                r, g, b = t[0], t[1], t[2]
                t = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)
            frames.append(t)
        tensor = torch.stack(frames, dim=0)  # (T, 1, H, W) or (T, H, W)
        tensor = tensor.squeeze(1) if tensor.shape[1] == 1 else tensor
        return (tensor / 255.0).to(device)

    # 3D: (H, W, C) -> (C, H, W)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))

    # 2D: (H, W) -> (1, H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]

    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    if tensor.shape[0] == 3:
        r, g, b = tensor[0], tensor[1], tensor[2]
        tensor = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)

    return (tensor / 255.0).to(device)




def build_environment(env_id: str, seed: int, frames: int, record_rgb: bool) -> gym.Env:
    renderer = "rgb_array" if record_rgb else None
    env = gym.make(env_id, render_mode=renderer, frame_skip=4)
    env = gym.wrappers.FrameStackObservation(env, frames)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate_agent(
    policy: nn.Module,
    args,
    device: torch.device,
    run_name: str,
) -> Tuple[float, List[np.ndarray]]:
    eval_env = build_environment(
        args.env_id,
        args.seed,
        args.frame_stack,
        record_rgb=args.capture_eval_video,
    )

    policy.eval()
    episode_scores: List[float] = []
    captured_frames: List[np.ndarray] = []

    for _ in range(args.eval_episodes):
        observation, _ = eval_env.reset()
        done = False
        cumulative_reward = 0.0
        stacked = prepare_input(observation, args.target_height, args.target_width, device).unsqueeze(0)

        while not done:
            with torch.no_grad():
                act = policy(stacked).argmax(dim=1).item()
            observation, reward, terminated, truncated, _ = eval_env.step(act)
            done = terminated or truncated
            cumulative_reward += float(reward)
            stacked = prepare_input(observation, args.target_height, args.target_width, device).unsqueeze(0)

            if args.capture_eval_video:
                try:
                    frame = eval_env.render()
                    captured_frames.append(frame)
                except Exception:
                    pass

        episode_scores.append(cumulative_reward)

    eval_env.close()
    policy.train()
    return float(np.mean(episode_scores)), captured_frames


def _write_line_plot(path: str, xs: Sequence[float], ys: Sequence[float], xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    marker = "." if len(xs) <= 200 else None
    plt.plot(xs, ys, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def export_charts(root: str, logs: HistoryBook) -> None:
    os.makedirs(root, exist_ok=True)

    if logs.losses:
        xs, ys = zip(*logs.losses)
        _write_line_plot(os.path.join(root, "loss.png"), xs, ys, "Step", "TD Loss", "Training TD Loss")

    if logs.epsilons:
        xs, ys = zip(*logs.epsilons)
        _write_line_plot(os.path.join(root, "epsilon.png"), xs, ys, "Step", "Epsilon", "Exploration Schedule")

    if logs.episodic:
        xs, returns, lengths = zip(*logs.episodic)
        _write_line_plot(os.path.join(root, "episode_returns.png"), xs, returns, "Step", "Return", "Episode Returns")
        _write_line_plot(os.path.join(root, "episode_lengths.png"), xs, lengths, "Step", "Length", "Episode Lengths")

    if logs.evaluations:
        xs, ys = zip(*logs.evaluations)
        _write_line_plot(os.path.join(root, "eval_returns.png"), xs, ys, "Step", "Avg Return", "Evaluation Returns")


def execute_training(args) -> None:
    if _vizdoom_registry is None:
        raise RuntimeError("vizdoom is not installed. Run `pip install vizdoom` before executing this script.")

    run_label = args.run_name or f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    os.makedirs(os.path.join("videos", run_label, "train"), exist_ok=True)
    os.makedirs(os.path.join("videos", run_label, "eval"), exist_ok=True)
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
    # checkpoint root is namespaced by run label
    ckpt_root = os.path.join(args.checkpoint_dir, run_label)
    if args.checkpoint_every > 0:
        os.makedirs(ckpt_root, exist_ok=True)

    env = build_environment(args.env_id, args.seed, args.frame_stack, record_rgb=args.capture_video)
    initial_obs, _ = env.reset()
    obs_shape = (args.frame_stack, args.target_height, args.target_width)

    action_space = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    Net = DuelingDoomQNetwork if getattr(args, "dueling", False) else DoomQNetwork
    policy = Net(args.frame_stack, action_space, args.target_height, args.target_width).to(device)
    target = Net(args.frame_stack, action_space, args.target_height, args.target_width).to(device)
    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        try:
            policy = torch.compile(policy)  # type: ignore[attr-defined]
            target = torch.compile(target)  # type: ignore[attr-defined]
        except Exception:
            pass
    target.load_state_dict(policy.state_dict())

    # Load model-only warm start if provided (overridden by resume checkpoints below)
    if args.load_model:
        state_dict = torch.load(args.load_model, map_location=device)
        policy.load_state_dict(state_dict)
        target.load_state_dict(state_dict)

    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    buffer_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
    replay = ReplayBuffer(args.buffer_size, buffer_space, env.action_space, device=device, handle_timeout_termination=False)

    logs = HistoryBook(losses=[], epsilons=[], episodic=[], evaluations=[])
    plot_root = os.path.join(args.plots_dir, run_label)

    def epsilon_schedule(step: int) -> float:
        limit = args.exploration_fraction * args.total_timesteps
        if limit <= 0:
            return args.end_e
        slope = (args.end_e - args.start_e) / limit
        return max(slope * step + args.start_e, args.end_e)

    # ----------------------
    # Resume support
    # ----------------------
    start_step = 0
    resume_path = ""
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        # auto-pick latest checkpoint in this run's checkpoint dir
        pattern = os.path.join(ckpt_root, "ckpt_*.pt")
        candidates = glob.glob(pattern)
        if candidates:
            # choose by step number in filename or mtime as fallback
            def _step_from_name(p: str) -> int:
                base = os.path.basename(p)
                try:
                    return int(base.split("_")[1].split(".")[0])
                except Exception:
                    return -1
            candidates.sort(key=lambda p: (_step_from_name(p), os.path.getmtime(p)))
            resume_path = candidates[-1]

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        # model/optimizer/target
        if "model" in ckpt:
            policy.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
        if "target" in ckpt:
            try:
                target.load_state_dict(ckpt["target"])  # type: ignore[arg-type]
            except Exception:
                target.load_state_dict(policy.state_dict())
        if "optim" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim"])  # type: ignore[arg-type]
            except Exception:
                pass
        # rng
        try:
            if "rng_python" in ckpt:
                random.setstate(ckpt["rng_python"])  # type: ignore[arg-type]
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])  # type: ignore[arg-type]
            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])  # type: ignore[arg-type]
            if torch.cuda.is_available() and "rng_cuda" in ckpt:
                try:
                    torch.cuda.set_rng_state_all(ckpt["rng_cuda"])  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            pass
        # next step
        try:
            start_step = int(ckpt.get("step", 0)) + 1
        except Exception:
            start_step = 0

    observation = initial_obs
    last_print = start_step
    progress = tqdm(total=args.total_timesteps, desc="Training", unit="step", initial=start_step)
    loop_start = time.time()

    def maybe_save_checkpoint(step: int) -> None:
        if args.checkpoint_every <= 0:
            return
        if step == 0 or step % args.checkpoint_every != 0:
            return
        payload = {
            "step": step,
            "model": policy.state_dict(),
            "target": target.state_dict(),
            "optim": optimizer.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                payload["rng_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        path = os.path.join(ckpt_root, f"ckpt_{step}.pt")
        try:
            torch.save(payload, path)
            # keep a copy named 'latest.pt' for convenience
            latest = os.path.join(ckpt_root, "latest.pt")
            try:
                import shutil
                shutil.copy2(path, latest)
            except Exception:
                pass
            if step % (args.checkpoint_every * 5) == 0:
                print(f"Checkpoint saved: {path}")
        except Exception as e:
            print(f"Warning: failed to save checkpoint at step {step}: {e}")

    for step in range(start_step, args.total_timesteps):
        state_tensor = prepare_input(observation, args.target_height, args.target_width, device)
        eps_value = epsilon_schedule(step)

        if random.random() < eps_value:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act = policy(state_tensor.unsqueeze(0)).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        next_state_tensor = prepare_input(next_obs, args.target_height, args.target_width, device)

        replay.add(
            state_tensor.cpu().numpy(),
            next_state_tensor.cpu().numpy(),
            np.array(act),
            np.array(reward),
            np.array(done),
            [info],
        )

        if step >= args.learning_starts and step % args.train_frequency == 0:
            batch = replay.sample(args.batch_size)
            with torch.no_grad():
                if getattr(args, "double_dqn", False):
                    # Double DQN: online net selects actions, target net evaluates
                    next_online = policy(batch.next_observations)
                    next_acts = next_online.argmax(dim=1, keepdim=True)
                    next_target = target(batch.next_observations)
                    next_values = next_target.gather(1, next_acts).squeeze(1)
                else:
                    next_values = target(batch.next_observations).max(dim=1)[0]
                targets = batch.rewards.flatten() + args.gamma * next_values * (1 - batch.dones.flatten())

            current = policy(batch.observations).gather(1, batch.actions).squeeze()
            optimizer.zero_grad(set_to_none=True)
            if getattr(args, "loss", "mse") == "huber":
                loss = nn.functional.smooth_l1_loss(current, targets)
            else:
                loss = nn.functional.mse_loss(current, targets)
            loss.backward()
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            logs.losses.append((step, loss.item()))
            logs.epsilons.append((step, eps_value))

        if step % args.target_network_frequency == 0:
            for source, dest in zip(policy.parameters(), target.parameters()):
                dest.data.copy_(args.tau * source.data + (1 - args.tau) * dest.data)

        if args.eval_freq > 0 and step > 0 and step % args.eval_freq == 0:
            avg_return, frames = evaluate_agent(policy, args, device, run_label)
            logs.evaluations.append((step, avg_return))
            print(f"Step {step}: eval avg return {avg_return:.2f}")
            if args.capture_eval_video and frames:
                filename = os.path.join("videos", run_label, "eval", f"eval_{step}.mp4")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                imageio.mimsave(filename, frames, fps=30)

        if "episode" in info:
            logs.episodic.append((step, info["episode"]["r"], info["episode"]["l"]))
            if step - last_print >= args.log_interval:
                elapsed = time.time() - loop_start
                sps = (step + 1) / max(elapsed, 1e-6)
                print(
                    f"Step={step} Return={info['episode']['r']:.1f} Length={info['episode']['l']} Eps={eps_value:.3f} SPS={sps:.1f}"
                )
                last_print = step
                progress.set_postfix({"eps": f"{eps_value:.2f}", "SPS": f"{sps:.1f}"})

        # periodic checkpointing
        maybe_save_checkpoint(step)

        observation = env.reset()[0] if done else next_obs
        progress.update(1)

    progress.close()
    total_time = time.time() - loop_start
    print(f"Training complete (took {total_time:.1f}s / {total_time/60:.1f}m). Running final eval...")

    final_avg, final_frames = evaluate_agent(policy, args, device, run_label)
    logs.evaluations.append((args.total_timesteps, final_avg))
    print(f"Final avg return: {final_avg:.2f}")

    if args.capture_eval_video and final_frames:
        final_video = os.path.join("videos", run_label, "eval", "final_eval.mp4")
        os.makedirs(os.path.dirname(final_video), exist_ok=True)
        imageio.mimsave(final_video, final_frames, fps=30)

    if args.save_model:
        model_path = os.path.join(args.save_dir, f"{run_label}.pth")
        torch.save(policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    export_charts(plot_root, logs)

    try:
        if (args.capture_video or args.capture_eval_video) and hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()
    except cv2.error:
        pass


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Improved VizDoom DQN trainer")
    parser.add_argument("--exp-name", type=str, default="improved-dqn-vizdoom")
    parser.add_argument("--run-name", type=str, default="", help="Override autogenerated run name")
    parser.add_argument("--env-id", type=str, default="VizdoomBasic-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target-network-frequency", type=int, default=50)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--target-height", type=int, default=84)
    parser.add_argument("--target-width", type=int, default=84)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=2000)
    # Learning improvements / toggles
    parser.add_argument("--double-dqn", action="store_true", help="Use Double DQN targets")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling Q-network head")
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help="TD loss function",
    )
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient norm clip (0 disables)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available for speed")
    parser.add_argument("--capture-video", dest="capture_video", action="store_true", default=True)
    parser.add_argument("--no-capture-video", dest="capture_video", action="store_false")
    parser.add_argument("--capture-eval-video", dest="capture_eval_video", action="store_true", default=True)
    parser.add_argument("--no-capture-eval-video", dest="capture_eval_video", action="store_false")
    parser.add_argument("--save-model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--no-save-model", dest="save_model", action="store_false")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--load-model", type=str, default="", help="Optional checkpoint to load before training/eval")
    # checkpoint/resume
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to store periodic training checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10000, help="How often to save checkpoints (in env steps). 0 disables")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint for this run name")
    parser.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint file (.pt)")
    return parser.parse_args()


if __name__ == "__main__":
    execute_training(parse_cli_args())
