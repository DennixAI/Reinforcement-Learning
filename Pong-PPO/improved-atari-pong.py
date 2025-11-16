import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import ale_py
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from tqdm import trange


gym.register_envs(ale_py)


@dataclass
class Config:
    exp_name: str = "PPO-Improved-Pong"
    env_id: str = "PongNoFrameskip-v4"
    seed: int = 11
    total_timesteps: int = 10_000_000
    num_envs: int = 8
    num_steps: int = 128
    num_minibatches: int = 4
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    clip_vloss: bool = True
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 2.5e-4
    lr_anneal: bool = True
    max_grad_norm: float = 0.5
    target_kl: float = 0.0
    frame_stack: int = 4
    eval_interval: int = 200
    num_eval_episodes: int = 5
    capture_video: bool = True
    save_final_video: bool = True
    video_folder: str = os.path.join("images", "ppo_improved")
    plot_folder: str = os.path.join("plots", "ppo_improved")
    use_amp: bool = False
    random_shift_pad: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RandomShiftsAug(nn.Module):
    """Implements the DrQ-style random shift augmentation."""

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad <= 0:
            return x
        n, c, h, w = x.shape
        padded = nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        eps = 2 * self.pad + 1
        coords = torch.randint(0, eps, size=(n, 2), device=x.device)
        crops = []
        for img, (y, x_shift) in zip(padded, coords):
            crops.append(img[:, y : y + h, x_shift : x_shift + w])
        return torch.stack(crops, dim=0)


class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, aug_pad: int):
        super().__init__()
        self.augment = RandomShiftsAug(aug_pad) if aug_pad > 0 else None
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            n_flatten = self.encoder(dummy).shape[1]
        self.policy = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy[-1].weight, 0.01)
        nn.init.zeros_(self.policy[-1].bias)

    def _maybe_augment(self, obs: torch.Tensor, use_aug: bool) -> torch.Tensor:
        if use_aug and self.augment is not None:
            return self.augment(obs)
        return obs

    def forward(self, obs: torch.Tensor, use_aug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self._maybe_augment(obs, use_aug)
        embedding = self.encoder(obs)
        logits = self.policy(embedding)
        value = self.value(embedding).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        use_aug: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, use_aug=use_aug)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, cfg: Config, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        self.cfg = cfg
        self.device = cfg.device
        self.obs = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape, device=self.device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs) + action_shape, device=self.device)
        self.logprobs = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.rewards = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.dones = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.values = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.advantages = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.returns = torch.zeros(cfg.num_steps, cfg.num_envs, device=self.device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.logprobs[self.step].copy_(log_prob)
        self.step = (self.step + 1) % self.cfg.num_steps

    def compute_returns(self, last_value: torch.Tensor, last_done: torch.Tensor) -> None:
        last_adv = torch.zeros_like(last_value)
        for t in reversed(range(self.cfg.num_steps)):
            if t == self.cfg.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + self.cfg.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_adv
        self.returns = self.advantages + self.values

    def get_minibatches(self) -> Iterable[Dict[str, torch.Tensor]]:
        batch_size = self.cfg.batch_size
        indices = torch.randperm(batch_size, device=self.device)
        obs = self.obs.reshape(batch_size, *self.obs.shape[2:])
        actions = self.actions.reshape(batch_size, *self.actions.shape[2:])
        logprobs = self.logprobs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        values = self.values.reshape(batch_size)

        for start in range(0, batch_size, self.cfg.minibatch_size):
            end = start + self.cfg.minibatch_size
            mb_inds = indices[start:end]
            yield {
                "obs": obs[mb_inds],
                "actions": actions[mb_inds],
                "logprobs": logprobs[mb_inds],
                "advantages": advantages[mb_inds],
                "returns": returns[mb_inds],
                "values": values[mb_inds],
            }


def make_env(env_id: str, seed: int, idx: int, capture_video: bool = False):
    def thunk():
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            terminal_on_life_loss=True,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True,
        )
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def prepare_obs(obs: np.ndarray, device: torch.device, frame_stack: int) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(0)
    if obs_tensor.shape[-1] == frame_stack:
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    elif obs_tensor.shape[1] == frame_stack and obs_tensor.ndim == 4:
        pass
    obs_tensor = obs_tensor / 255.0 if obs_tensor.max().item() > 1.0 else obs_tensor
    return obs_tensor


def evaluate_policy(
    agent: ActorCritic,
    cfg: Config,
    run_name: str,
    record: bool = False,
) -> Tuple[List[float], List[np.ndarray]]:
    env = make_env(cfg.env_id, cfg.seed + 10_000, 0, capture_video=record)()
    obs, _ = env.reset(seed=cfg.seed + 10_000)
    frames: List[np.ndarray] = []
    returns: List[float] = []
    agent.eval()
    ep_reward = 0.0
    while len(returns) < cfg.num_eval_episodes:
        obs_tensor = prepare_obs(obs, cfg.device, cfg.frame_stack)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True, use_aug=False)
        obs, reward, terminated, truncated, info = env.step(action.cpu().item())
        ep_reward += float(reward)
        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if terminated or truncated:
            returns.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
    env.close()
    agent.train()
    return returns, frames


def save_plots(
    cfg: Config,
    steps: List[int],
    policy_losses: List[float],
    value_losses: List[float],
    entropy_losses: List[float],
    train_returns: List[float],
    eval_steps: List[int],
    eval_returns: List[float],
) -> None:
    if not steps:
        return
    os.makedirs(cfg.plot_folder, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.plot(steps, policy_losses, label="Policy")
    ax.plot(steps, value_losses, label="Value")
    ax.plot(steps, entropy_losses, label="Entropy")
    ax.set_title("Improved PPO Losses")
    ax.set_xlabel("Environment steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_folder, "improved_pong_losses.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(steps, train_returns, label="Train reward")
    if eval_returns:
        ax.plot(eval_steps, eval_returns, marker="o", linestyle="--", label="Eval return")
    ax.set_title("Improved PPO Returns")
    ax.set_xlabel("Environment steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_folder, "improved_pong_returns.png"))
    plt.close(fig)


def main():
    cfg = Config()
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    os.makedirs(cfg.video_folder, exist_ok=True)
    os.makedirs(cfg.plot_folder, exist_ok=True)
    set_seed(cfg.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed, idx, capture_video=False) for idx in range(cfg.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    sample_obs, _ = envs.reset(seed=cfg.seed)
    obs_tensor = prepare_obs(sample_obs, cfg.device, cfg.frame_stack)
    obs_shape = obs_tensor.shape[1:]

    agent = ActorCritic(envs.single_action_space.n, cfg.random_shift_pad).to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    amp_enabled = cfg.use_amp and cfg.device.type == "cuda"
    if amp_enabled:
        try:
            scaler = amp.GradScaler(device_type=cfg.device.type, enabled=True)
            autocast_ctx = lambda: amp.autocast(device_type=cfg.device.type, enabled=True)
        except TypeError:
            from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast

            scaler = CudaGradScaler(enabled=True)
            autocast_ctx = lambda: cuda_autocast(enabled=True)
    else:
        scaler = None
        autocast_ctx = nullcontext
    buffer = RolloutBuffer(cfg, obs_shape, action_shape)

    global_step = 0
    next_obs = obs_tensor
    next_done = torch.zeros(cfg.num_envs, device=cfg.device)
    start_time = time.time()
    num_updates = cfg.total_timesteps // cfg.batch_size

    training_steps: List[int] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropy_losses: List[float] = []
    running_train_returns: List[float] = []
    eval_steps: List[int] = []
    eval_returns: List[float] = []
    episodic_returns: List[float] = []

    for update in trange(1, num_updates + 1, desc="Improved PPO Updates"):
        agent.train()
        if cfg.lr_anneal:
            frac = 1.0 - (update - 1) / max(1, num_updates)
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(
                    next_obs, use_aug=True, deterministic=False
                )
            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=cfg.device)
            buffer.add(next_obs, action, reward_tensor, next_done, value, logprob)

            next_obs = prepare_obs(next_obs_np, cfg.device, cfg.frame_stack)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=cfg.device)

            if "final_observation" in info:
                for idx, final_obs in enumerate(info["final_observation"]):
                    final_reward = info.get("final_reward", [None])[idx]
                    if final_obs is not None and final_reward is not None:
                        episodic_returns.append(float(final_reward))
            if "final_info" in info:
                for final_info in info["final_info"]:
                    if final_info and "episode" in final_info:
                        episodic_returns.append(final_info["episode"]["r"])

        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs, use_aug=False)[-1]
        buffer.compute_returns(next_value, next_done)

        pg_loss, v_loss, entropy_loss = 0.0, 0.0, 0.0
        approx_kl = 0.0
        clip_fraction = 0.0

        for epoch in range(cfg.ppo_epochs):
            for batch in buffer.get_minibatches():
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        batch["obs"], action=batch["actions"].long(), use_aug=True
                    )
                    ratio = torch.exp(new_logprob - batch["logprobs"])
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss_batch = torch.max(pg_loss1, pg_loss2).mean()

                    value_pred = new_value
                    if cfg.clip_vloss:
                        v_loss_unclipped = (value_pred - batch["returns"]) ** 2
                        v_clipped = batch["values"] + torch.clamp(
                            value_pred - batch["values"], -cfg.clip_coef, cfg.clip_coef
                        )
                        v_loss_clipped = (v_clipped - batch["returns"]) ** 2
                        v_loss_batch = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss_batch = 0.5 * ((batch["returns"] - value_pred) ** 2).mean()

                    entropy_batch = entropy.mean()
                    loss = pg_loss_batch + cfg.value_coef * v_loss_batch - cfg.entropy_coef * entropy_batch
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                pg_loss += pg_loss_batch.item()
                v_loss += v_loss_batch.item()
                entropy_loss += entropy_batch.item()
                with torch.no_grad():
                    log_ratio = batch["logprobs"] - new_logprob
                    approx_kl += torch.mean(log_ratio).item()
                    clip_fraction += torch.mean((torch.abs(ratio - 1.0) > cfg.clip_coef).float()).item()
            if cfg.target_kl and approx_kl / (epoch + 1) > cfg.target_kl:
                break

        training_steps.append(global_step)
        policy_losses.append(pg_loss / (cfg.ppo_epochs * cfg.num_minibatches))
        value_losses.append(v_loss / (cfg.ppo_epochs * cfg.num_minibatches))
        entropy_losses.append(entropy_loss / (cfg.ppo_epochs * cfg.num_minibatches))
        running_train_returns.append(np.mean(episodic_returns[-10:]) if episodic_returns else 0.0)

        if update % cfg.eval_interval == 0:
            eval_ret, _ = evaluate_policy(agent, cfg, run_name, record=False)
            eval_steps.append(global_step)
            eval_returns.append(float(np.mean(eval_ret)))

    duration = time.time() - start_time
    print(f"Training completed in {duration / 3600:.2f} hours.")

    if cfg.save_final_video:
        eval_ret, eval_frames = evaluate_policy(agent, cfg, run_name, record=True)
        if eval_frames:
            video_path = os.path.join(cfg.video_folder, f"improved_pong_eval_{run_name}.mp4")
            imageio.mimsave(video_path, eval_frames, fps=30)
            print(f"Saved evaluation video to {video_path} (avg return {np.mean(eval_ret):.2f})")

    save_plots(
        cfg,
        training_steps,
        policy_losses,
        value_losses,
        entropy_losses,
        running_train_returns,
        eval_steps,
        eval_returns,
    )
    envs.close()


if __name__ == "__main__":
    main()
