"""
PPO Baseline on MiniHack — gymnasium API compatible
====================================================
Usage:
    python ppo_baseline_v2.py --env MiniHack-Room-5x5-v0 --steps 100000
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import minihack  # noqa: registers MiniHack envs
from nle import nethack

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
MAX_GRAD     = 0.5
LR           = 2.5e-4
ROLLOUT_LEN  = 128
N_EPOCHS     = 4
BATCH_SIZE   = 32
OBS_KEY      = "glyphs"


# ── Network ───────────────────────────────────────────────────────────────────
class GlyphEncoder(nn.Module):
    def __init__(self, n_glyphs=nethack.MAX_GLYPH, embed_dim=16, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(n_glyphs, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),        nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, hidden), nn.ReLU())
        self.hidden = hidden

    def forward(self, g):
        x = self.embed(g).permute(0, 3, 1, 2)
        return self.fc(self.conv(x).flatten(1))


class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.encoder = GlyphEncoder()
        h = self.encoder.hidden
        self.policy = nn.Linear(h, n_actions)
        self.value  = nn.Linear(h, 1)

    def forward(self, g):
        f = self.encoder(g)
        return self.policy(f), self.value(f).squeeze(-1)

    def get_action_and_value(self, g, action=None):
        logits, val = self(g)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val


# ── Environment ───────────────────────────────────────────────────────────────
def make_env(env_id):
    return gym.make(
        env_id,
        observation_keys=("glyphs", "blstats"),
        reward_win=1.0,
        reward_lose=-1.0,
        penalty_step=-0.001,
    )


# ── Rollout buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n, obs_shape, device):
        self.device   = device
        self.glyphs   = torch.zeros((n, *obs_shape), dtype=torch.long).to(device)
        self.actions  = torch.zeros(n, dtype=torch.long).to(device)
        self.logprobs = torch.zeros(n).to(device)
        self.rewards  = torch.zeros(n).to(device)
        self.dones    = torch.zeros(n).to(device)
        self.values   = torch.zeros(n).to(device)
        self.ptr = 0

    def add(self, g, a, lp, r, d, v):
        i = self.ptr
        self.glyphs[i]   = g
        self.actions[i]  = a
        self.logprobs[i] = lp
        self.rewards[i]  = r
        self.dones[i]    = d
        self.values[i]   = v
        self.ptr += 1

    def compute_returns(self, last_v, gamma, lam):
        adv = torch.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.ptr)):
            nv  = last_v if t == self.ptr - 1 else self.values[t + 1]
            nnt = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nv * nnt - self.values[t]
            gae   = delta + gamma * lam * nnt * gae
            adv[t] = gae
        return adv + self.values, adv

    def reset(self):
        self.ptr = 0


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, buf, last_v, device):
    rets, adv = buf.compute_returns(last_v, GAMMA, GAE_LAMBDA)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(N_EPOCHS):
        idx = np.random.permutation(buf.ptr)
        for s in range(0, buf.ptr, BATCH_SIZE):
            i = torch.tensor(idx[s:s + BATCH_SIZE], device=device)
            _, lp, ent, v = model.get_action_and_value(buf.glyphs[i], buf.actions[i])
            r    = (lp - buf.logprobs[i]).exp()
            loss = (
                -torch.min(r * adv[i],
                           torch.clamp(r, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[i]).mean()
                + VALUE_COEF * 0.5 * (v - rets[i]).pow(2).mean()
                - ENTROPY_COEF * ent.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            optimizer.step()


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = make_env(args.env)
    n_actions = env.action_space.n
    obs_shape = env.observation_space["glyphs"].shape
    print(f"Env: {args.env} | Actions: {n_actions} | Obs: {obs_shape}")

    model     = ActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    buf       = RolloutBuffer(ROLLOUT_LEN, obs_shape, device)

    obs, _  = env.reset()
    ep_ret  = 0.0
    ep_count= 0
    total_steps = 0
    episode_returns = []
    done = False
    start = time.time()

    print(f"Training for {args.steps} steps...\n")

    while total_steps < args.steps:
        buf.reset()

        for _ in range(ROLLOUT_LEN):
            g = torch.tensor(obs[OBS_KEY], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                a, lp, _, v = model.get_action_and_value(g)

            next_obs, r, terminated, truncated, info = env.step(a.item())
            done = terminated or truncated
            ep_ret += r
            total_steps += 1

            buf.add(g.squeeze(0), a, lp, r, float(done), v)
            obs = next_obs

            if done:
                ep_count += 1
                episode_returns.append(ep_ret)
                if ep_count % 10 == 0:
                    mean_ret = np.mean(episode_returns[-10:])
                    sps      = total_steps / (time.time() - start)
                    print(f"Steps: {total_steps:7d} | Eps: {ep_count:5d} | "
                          f"MeanRet(10): {mean_ret:7.3f} | SPS: {sps:.0f}")
                ep_ret = 0.0
                obs, _ = env.reset()
                done = False

        g_last = torch.tensor(obs[OBS_KEY], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, lv = model.get_action_and_value(g_last)
        ppo_update(model, optimizer, buf, lv.item() if not done else 0.0, device)

    env.close()
    final = np.mean(episode_returns[-20:]) if episode_returns else 0.0
    print(f"\nTraining complete. Final mean return (last 20): {final:.3f}")
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",       default="MiniHack-Room-5x5-v0")
    parser.add_argument("--steps",     type=int, default=100_000)
    parser.add_argument("--save_path", default="ppo_baseline.pt")
    train(parser.parse_args())
