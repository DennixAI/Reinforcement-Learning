import os
import time
import random
import csv
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import imageio

from walker_core import Policy, make_thunk, evaluate


SEED = 123
ENV_ID = "BipedalWalker-v3"

# Throughput settings (match original behavior)
NUM_ENVS = 8
ROLLOUT_STEPS = 128
TOTAL_TIMESTEPS = 5_000_000

# PPO settings
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
MINIBATCHES = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GRAD_NORM_CLIP = 0.5

# Misc
OBS_CLIP = 10.0
CAPTURE_VIDEO = False
EVAL_EVERY = 100
EVAL_EPISODES = 5

# Checkpointing
SAVE_EVERY = 1000
CKPT_DIR = "checkpoints_walker"

# Final video after training
SAVE_FINAL_VIDEO = True
FINAL_VIDEO_EPISODES = 3
FINAL_VIDEO_FPS = 30
VIDEO_DIR = "videos"

# Visualization
PLOTS_DIR = "plots"


def _write_metrics_header(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "update",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "fps",
            "eval_avg_return",
        ])


def _append_metric(path: str, row: List[float]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    env_fns = [make_thunk(ENV_ID, SEED, i, gamma=GAMMA, obs_clip=OBS_CLIP) for i in range(NUM_ENVS)]
    envs = gym.vector.SyncVectorEnv(env_fns)

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous actions required"

    policy = Policy(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

    obs_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, obs_dim), dtype=torch.float32, device=device)
    act_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, act_dim), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=torch.float32, device=device)
    done_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=torch.float32, device=device)
    val_buf = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=torch.float32, device=device)

    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(NUM_ENVS, dtype=torch.float32, device=device)

    BATCH_SIZE = NUM_ENVS * ROLLOUT_STEPS
    MINIBATCH_SIZE = BATCH_SIZE // MINIBATCHES
    NUM_UPDATES = TOTAL_TIMESTEPS // BATCH_SIZE

    metrics: List[Dict[str, float]] = []
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics_csv_path = os.path.join(PLOTS_DIR, "metrics_walker.csv")
    _write_metrics_header(metrics_csv_path)

    start = time.time()
    for upd in range(1, NUM_UPDATES + 1):
        for t in range(ROLLOUT_STEPS):
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                action, logp, _, value = policy.act(next_obs)

            val_buf[t] = value
            act_buf[t] = action
            logp_buf[t] = logp

            nobs, rew, term, trunc, _ = envs.step(action.detach().cpu().numpy())
            done = np.logical_or(term, trunc)

            rew_buf[t] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, next_val = policy.forward(next_obs)
            advantages = torch.zeros_like(rew_buf, device=device)
            last_gae = torch.zeros(NUM_ENVS, dtype=torch.float32, device=device)
            for t in reversed(range(ROLLOUT_STEPS)):
                mask = 1.0 - done_buf[t]
                delta = rew_buf[t] + GAMMA * next_val * mask - val_buf[t]
                last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
                advantages[t] = last_gae
                next_val = val_buf[t]
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(BATCH_SIZE, obs_dim)
        b_acts = act_buf.reshape(BATCH_SIZE, act_dim)
        b_logp = logp_buf.reshape(BATCH_SIZE)
        b_ret = returns.reshape(BATCH_SIZE)
        b_adv = advantages.reshape(BATCH_SIZE)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        idx = torch.randperm(BATCH_SIZE, device=device)
        for _ in range(EPOCHS):
            idx = torch.randperm(BATCH_SIZE, device=device)
            for start_i in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb = idx[start_i : start_i + MINIBATCH_SIZE]
                mb_obs = b_obs[mb]
                mb_act = b_acts[mb]
                mb_old_logp = b_logp[mb]
                mb_adv = b_adv[mb]
                mb_ret = b_ret[mb]

                new_logp, ent, value = policy.evaluate(mb_obs, mb_act)
                ratio = (new_logp - mb_old_logp).exp()
                pg_loss1 = mb_adv * ratio
                pg_loss2 = mb_adv * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
                value_loss = VALUE_COEF * nn.functional.mse_loss(value, mb_ret)
                loss = policy_loss + value_loss - ENTROPY_COEF * ent.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM_CLIP)
                optimizer.step()

        elapsed = time.time() - start
        fps = (upd * BATCH_SIZE) / max(1e-9, elapsed)

        eval_avg_return = None
        if upd % EVAL_EVERY == 0:
            eval_avg_return = evaluate(
                policy,
                device,
                envs,
                env_id=ENV_ID,
                seed=SEED,
                eval_episodes=EVAL_EPISODES,
                obs_clip=OBS_CLIP,
            )
            print(f"eval @ update {upd}: avg return = {eval_avg_return:.2f}")

        if upd % 10 == 0:
            print(
                f"update {upd}/{NUM_UPDATES} | loss {loss.item():.4f} | pi {policy_loss.item():.4f} | v {value_loss.item():.4f} | fps ~{fps:,.0f}"
            )

        try:
            entropy_val = float(ent.mean().item())
        except Exception:
            entropy_val = float("nan")

        metrics_row = {
            "update": upd,
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": entropy_val,
            "fps": float(fps),
            "eval_avg_return": float(eval_avg_return) if eval_avg_return is not None else float("nan"),
        }
        metrics.append(metrics_row)
        _append_metric(
            metrics_csv_path,
            [
                metrics_row["update"],
                metrics_row["loss"],
                metrics_row["policy_loss"],
                metrics_row["value_loss"],
                metrics_row["entropy"],
                metrics_row["fps"],
                metrics_row["eval_avg_return"],
            ],
        )

        if upd % SAVE_EVERY == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            ckpt_path = os.path.join(CKPT_DIR, f"ppo_{ENV_ID}_upd{upd}.pt")
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": upd,
                    "seed": SEED,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

    obs_rms = None
    try:
        obs_rms = envs.get_attr("obs_rms")[0]
    except Exception:
        obs_rms = None

    envs.close()

    if SAVE_FINAL_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(SEED)

        frames = []
        policy.eval()
        with torch.no_grad():
            for ep in range(FINAL_VIDEO_EPISODES):
                obs, _ = env.reset(seed=SEED + ep)
                done = False
                while not done:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                    if obs_rms is not None:
                        norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -OBS_CLIP, OBS_CLIP)
                    else:
                        norm_obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)

                    obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    mu, std, _ = policy.forward(obs_t)
                    action = mu.squeeze(0).cpu().numpy()
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

        ts = int(time.time())
        out_path = os.path.join(VIDEO_DIR, f"walker_final_{ENV_ID}_{ts}.mp4")
        if len(frames) > 0:
            imageio.mimsave(out_path, frames, fps=FINAL_VIDEO_FPS, codec="libx264")
            print(f"final video saved: {out_path}")
        env.close()

    try:
        from viz import save_plots

        save_plots(metrics, out_dir=PLOTS_DIR)
    except Exception as e:
        print(f"plotting failed: {e}")

    print("Training complete.")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
