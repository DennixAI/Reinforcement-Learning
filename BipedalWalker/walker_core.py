import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


def _layer_init(m: nn.Module, std: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(m.weight, std)
    nn.init.constant_(m.bias, bias)
    return m


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor_fc1 = _layer_init(nn.Linear(obs_dim, 256))
        self.actor_fc2 = _layer_init(nn.Linear(256, 256))
        self.actor_fc3 = _layer_init(nn.Linear(256, 16))
        self.actor_mu = _layer_init(nn.Linear(16, act_dim))

        self.critic_fc1 = _layer_init(nn.Linear(obs_dim, 512))
        self.critic_fc2 = _layer_init(nn.Linear(512, 512))
        self.critic_fc3 = _layer_init(nn.Linear(512, 256))
        self.critic_v = _layer_init(nn.Linear(256, 1), std=1.0)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_a = torch.tanh(self.actor_fc1(x))
        x_a = torch.tanh(self.actor_fc2(x_a))
        x_a = torch.tanh(self.actor_fc3(x_a))
        mu = self.actor_mu(x_a)
        std = torch.nn.functional.softplus(self.log_std) + 1e-5
        std = std.expand_as(mu)
        x_v = torch.tanh(self.critic_fc1(x))
        x_v = torch.tanh(self.critic_fc2(x_v))
        x_v = torch.tanh(self.critic_fc3(x_v))
        value = self.critic_v(x_v).squeeze(-1)
        return mu, std, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, dist.entropy().sum(-1), value

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, value


def make_thunk(
    env_id: str,
    seed: int,
    idx: int,
    gamma: float,
    obs_clip: float = 10.0,
    eval_mode: bool = False,
    render_mode=None,
) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        if not eval_mode:
            env = gym.wrappers.TransformObservation(
                env,
                lambda o: np.clip(o, -abs(obs_clip), abs(obs_clip)),
                env.observation_space,
            )
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -abs(obs_clip), abs(obs_clip)))

        env.action_space.seed(seed + idx)
        try:
            env.seed(seed + idx)
        except Exception:
            pass
        return env

    return _thunk


@torch.no_grad()
def evaluate(
    policy: Policy,
    device: torch.device,
    train_envs: gym.vector.VectorEnv,
    env_id: str,
    seed: int,
    eval_episodes: int,
    obs_clip: float = 10.0,
) -> float:
    eval_env = make_thunk(env_id, seed, 0, gamma=1.0, obs_clip=obs_clip, eval_mode=True)()
    eval_env.action_space.seed(seed)

    obs_rms = None
    try:
        obs_rms = train_envs.get_attr("obs_rms")[0]
    except Exception:
        obs_rms = None

    policy.eval()
    scores = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_r = 0.0
        while not done:
            if obs_rms is not None:
                ob = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -abs(obs_clip), abs(obs_clip))
            else:
                ob = np.clip(obs, -abs(obs_clip), abs(obs_clip))
            ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std, _ = policy.forward(ob_t)
            action = torch.distributions.Normal(mu, std).mean
            obs, r, term, trunc, _ = eval_env.step(action.squeeze(0).cpu().numpy())
            done = term or trunc
            total_r += r
        scores.append(total_r)

    eval_env.close()
    policy.train()
    return float(np.mean(scores))


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

    env_fns = [make_thunk(ENV_ID, SEED, i) for i in range(NUM_ENVS)]
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

    # metrics + csv setup
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics = []
    metrics_csv_path = os.path.join(PLOTS_DIR, "metrics_walker.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", newline="") as f:
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

            nobs, rew, term, trunc, info = envs.step(action.detach().cpu().numpy())
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

        # compute instantaneous throughput
        elapsed = time.time() - start
        fps = (upd * BATCH_SIZE) / max(1e-9, elapsed)

        # optional eval
        eval_avg_return = None
        if upd % EVAL_EVERY == 0:
            eval_avg_return = evaluate(policy, device, envs)
            print(f"eval @ update {upd}: avg return = {eval_avg_return:.2f}")

        # log progress every 10 updates
        if upd % 10 == 0:
            print(
                f"update {upd}/{NUM_UPDATES} | loss {loss.item():.4f} | pi {policy_loss.item():.4f} | v {value_loss.item():.4f} | fps ~{fps:,.0f}"
            )

        # record metrics
        try:
            entropy_val = float(getattr(ent, "mean", lambda: ent)().item() if hasattr(ent, "mean") else float(ent))
        except Exception:
            entropy_val = float("nan")

        m = {
            "update": upd,
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": entropy_val,
            "fps": float(fps),
            "eval_avg_return": (float(eval_avg_return) if eval_avg_return is not None else float("nan")),
        }
        metrics.append(m)
        # append to CSV for durability
        with open(metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                m["update"],
                m["loss"],
                m["policy_loss"],
                m["value_loss"],
                m["entropy"],
                m["fps"],
                m["eval_avg_return"],
            ])

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
                        norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                    else:
                        norm_obs = np.clip(obs, -10, 10)

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

    # plots
    try:
        from viz import save_plots

        save_plots(metrics, out_dir=PLOTS_DIR)
    except Exception as e:
        print(f"plotting failed: {e}")

    print("Training complete.")
