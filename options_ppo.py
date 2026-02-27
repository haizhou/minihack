"""
Option-level PPO on MiniHack — gymnasium API compatible
========================================================
Meta-policy selects which Option to run. Each Option is a hardcoded
macro-action that executes primitive steps internally.

Options:
  0 - ExploreOption          : random walk (up to MAX_OPTION_STEPS steps)
  1 - NavigateToStaircase    : BFS toward '>' tile
  2 - PickupItem             : BFS toward nearest item, then pick up

Usage:
    python options_ppo_v2.py --env MiniHack-Room-5x5-v0 --steps 100000
"""
import argparse
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import minihack  # noqa
from nle import nethack

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_EPS         = 0.2
ENTROPY_COEF     = 0.01
VALUE_COEF       = 0.5
MAX_GRAD         = 0.5
LR               = 2.5e-4
ROLLOUT_LEN      = 64
N_EPOCHS         = 4
BATCH_SIZE       = 16
MAX_OPTION_STEPS = 30
OBS_KEY          = "glyphs"

# Primitive compass actions (NLE default action indices)
COMPASS = {(-1,0):0, (0,1):1, (1,0):2, (0,-1):3,
           (-1,1):4, (1,1):5, (1,-1):6, (-1,-1):7}
PICKUP_ACTION = 44
WAIT_ACTION   = 17

STAIRCASE_CHAR = ord('>')
ITEM_CHARS     = set(ord(c) for c in r'!"$%&' + r"'()*+,-./:;<=>?@[\]^_`{|}~")


# ── Glyph helpers ─────────────────────────────────────────────────────────────
def get_agent_pos(blstats):
    return int(blstats[1]), int(blstats[0])   # row, col


def find_chars(glyphs, target_chars):
    positions = []
    for r in range(glyphs.shape[0]):
        for c in range(glyphs.shape[1]):
            try:
                if nethack.glyph_to_ascii(glyphs[r, c]) in target_chars:
                    positions.append((r, c))
            except Exception:
                pass
    return positions


def bfs_next_action(glyphs, blstats, target_chars):
    h, w   = glyphs.shape
    start  = get_agent_pos(blstats)
    targets= set(find_chars(glyphs, target_chars))
    if not targets:
        return None
    visited = {start}
    queue   = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        if pos in targets:
            if not path:
                return None
            dr = max(-1, min(1, path[0][0] - start[0]))
            dc = max(-1, min(1, path[0][1] - start[1]))
            return COMPASS.get((dr, dc), WAIT_ACTION)
        r, c = pos
        for dr, dc in COMPASS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


# ── Options ───────────────────────────────────────────────────────────────────
class Option:
    name = "BaseOption"
    def initiate(self, obs): pass
    def act(self, obs) -> int: raise NotImplementedError
    def terminate(self, obs, step) -> bool: raise NotImplementedError


class ExploreOption(Option):
    name = "Explore"
    def initiate(self, obs): pass
    def act(self, obs):
        return int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        return step >= MAX_OPTION_STEPS


class NavigateToStaircaseOption(Option):
    name = "NavigateToStaircase"
    def initiate(self, obs): pass
    def act(self, obs):
        a = bfs_next_action(obs["glyphs"], obs["blstats"], {STAIRCASE_CHAR})
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        if step >= MAX_OPTION_STEPS:
            return True
        r, c = get_agent_pos(obs["blstats"])
        return (r, c) in set(find_chars(obs["glyphs"], {STAIRCASE_CHAR}))


class PickupItemOption(Option):
    name = "PickupItem"
    def initiate(self, obs):
        self._picked = False
    def act(self, obs):
        r, c = get_agent_pos(obs["blstats"])
        items = find_chars(obs["glyphs"], ITEM_CHARS)
        if (r, c) in items:
            self._picked = True
            return PICKUP_ACTION
        a = bfs_next_action(obs["glyphs"], obs["blstats"], ITEM_CHARS)
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        return step >= MAX_OPTION_STEPS or self._picked


OPTIONS   = [ExploreOption(), NavigateToStaircaseOption(), PickupItemOption()]
N_OPTIONS = len(OPTIONS)


# ── Option execution ──────────────────────────────────────────────────────────
def execute_option(env, obs, option):
    option.initiate(obs)
    cum_r = 0.0
    step  = 0
    done  = False
    info  = {}
    while not option.terminate(obs, step):
        a = option.act(obs)
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        cum_r += r
        step  += 1
        if done:
            break
    return obs, cum_r, done, step, info


# ── Network ───────────────────────────────────────────────────────────────────
class GlyphEncoder(nn.Module):
    def __init__(self, n_glyphs=nethack.MAX_GLYPH, embed_dim=16, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(n_glyphs, embed_dim)
        self.conv  = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),        nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, hidden), nn.ReLU())
        self.hidden = hidden

    def forward(self, g):
        return self.fc(self.conv(self.embed(g).permute(0, 3, 1, 2)).flatten(1))


class MetaActorCritic(nn.Module):
    def __init__(self, n_options):
        super().__init__()
        self.encoder = GlyphEncoder()
        h = self.encoder.hidden
        self.policy = nn.Linear(h, n_options)
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


# ── Rollout buffer (option-level) ─────────────────────────────────────────────
class OptionBuffer:
    def __init__(self, n, obs_shape, device):
        self.device   = device
        self.glyphs   = torch.zeros((n, *obs_shape), dtype=torch.long).to(device)
        self.options  = torch.zeros(n, dtype=torch.long).to(device)
        self.logprobs = torch.zeros(n).to(device)
        self.rewards  = torch.zeros(n).to(device)
        self.dones    = torch.zeros(n).to(device)
        self.values   = torch.zeros(n).to(device)
        self.ptr = 0

    def add(self, g, o, lp, r, d, v):
        i = self.ptr
        self.glyphs[i]=g; self.options[i]=o; self.logprobs[i]=lp
        self.rewards[i]=r; self.dones[i]=d; self.values[i]=v
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

    def reset(self): self.ptr = 0


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, buf, last_v, device):
    rets, adv = buf.compute_returns(last_v, GAMMA, GAE_LAMBDA)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(N_EPOCHS):
        idx = np.random.permutation(buf.ptr)
        for s in range(0, buf.ptr, BATCH_SIZE):
            i = torch.tensor(idx[s:s + BATCH_SIZE], device=device)
            _, lp, ent, v = model.get_action_and_value(buf.glyphs[i], buf.options[i])
            r    = (lp - buf.logprobs[i]).exp()
            loss = (
                -torch.min(r * adv[i],
                           torch.clamp(r, 1-CLIP_EPS, 1+CLIP_EPS) * adv[i]).mean()
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
    print(f"Options: {[o.name for o in OPTIONS]}\n")

    env = gym.make(args.env, observation_keys=("glyphs","blstats"),
                   reward_win=1.0, reward_lose=-1.0, penalty_step=-0.001)
    obs_shape = env.observation_space["glyphs"].shape

    model     = MetaActorCritic(N_OPTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    buf       = OptionBuffer(ROLLOUT_LEN, obs_shape, device)

    obs, _     = env.reset()
    ep_ret     = 0.0
    ep_count   = 0
    total_prim = 0
    done       = False
    episode_returns  = []
    option_counts    = np.zeros(N_OPTIONS, dtype=int)
    start = time.time()

    print(f"Training Option-PPO on {args.env} for ~{args.steps} primitive steps...\n")

    while total_prim < args.steps:
        buf.reset()

        for _ in range(ROLLOUT_LEN):
            g = torch.tensor(obs[OBS_KEY], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                opt_idx, lp, _, v = model.get_action_and_value(g)

            option = OPTIONS[opt_idx.item()]
            option_counts[opt_idx.item()] += 1

            obs, cum_r, done, prim_n, _ = execute_option(env, obs, option)
            ep_ret     += cum_r
            total_prim += prim_n

            buf.add(g.squeeze(0), opt_idx, lp, cum_r, float(done), v)

            if done:
                ep_count += 1
                episode_returns.append(ep_ret)
                if ep_count % 10 == 0:
                    mean_ret = np.mean(episode_returns[-10:])
                    sps      = total_prim / (time.time() - start)
                    counts   = " | ".join(f"{OPTIONS[i].name}:{option_counts[i]}"
                                          for i in range(N_OPTIONS))
                    print(f"PrimSteps:{total_prim:7d} | Eps:{ep_count:5d} | "
                          f"MeanRet:{mean_ret:7.3f} | SPS:{sps:.0f}\n  [{counts}]")
                ep_ret = 0.0
                obs, _ = env.reset()
                done = False

            if total_prim >= args.steps:
                break

        g_last = torch.tensor(obs[OBS_KEY], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, lv = model.get_action_and_value(g_last)
        ppo_update(model, optimizer, buf, lv.item() if not done else 0.0, device)

    env.close()
    final = np.mean(episode_returns[-20:]) if episode_returns else 0.0
    print(f"\nTraining complete. Final mean return (last 20): {final:.3f}")
    print("\nOption usage:")
    for i, o in enumerate(OPTIONS):
        print(f"  {o.name:30s}: {option_counts[i]:6d} times")
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",       default="MiniHack-Room-5x5-v0")
    parser.add_argument("--steps",     type=int, default=100_000)
    parser.add_argument("--save_path", default="options_ppo.pt")
    train(parser.parse_args())
