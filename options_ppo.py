"""
Option-level PPO on MiniHack — gymnasium API compatible
========================================================
Meta-policy selects which Option to run. Each Option is a hardcoded
macro-action that executes primitive steps internally.

Options:
  0 - ExploreOption          : random walk (up to MAX_OPTION_STEPS steps)
  1 - NavigateToStaircaseOption : BFS toward '>' tile
  2 - PickupItemOption       : BFS toward nearest item, then pick up
  3 - FindKeyOption          : BFS toward '(' key glyph (KeyRoom specific)
  4 - OpenDoorOption         : BFS toward '+' door, apply key to open it

Usage:
    python options_ppo.py --env MiniHack-Room-5x5-v0 --steps 100000
    python options_ppo.py --env MiniHack-KeyRoom-S5-v0 --steps 200000
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
MAX_OPTION_STEPS = 50   # increased for harder envs
OBS_KEY          = "glyphs"

# Intrinsic reward scale; per-sub-goal magnitude = ETA * P_node * decay
# P_node comes from kg_planner.get_node_probs; decay matches kg_bias schedule
ETA_REWARD = 0.5

# Primitive compass actions (NLE default action indices)
COMPASS = {(-1,0):0, (0,1):1, (1,0):2, (0,-1):3,
           (-1,1):4, (1,1):5, (1,-1):6, (-1,-1):7}
PICKUP_ACTION = 8    # index of Command.PICKUP in KeyRoom action space
APPLY_ACTION  = 9    # index of Command.APPLY in KeyRoom action space
WAIT_ACTION   = 0    # fallback: move North

# Glyph ASCII targets
STAIRCASE_CHAR = ord('>')
KEY_CHAR       = ord('(')        # key glyph in NetHack
DOOR_CHARS     = {ord('+')}  # closed door — '+' only, '|' and '-' are walls
ITEM_CHARS     = set(ord(c) for c in r'!"$%&*')  # only real items, excludes floor/walls/key/door


# ── Glyph helpers ─────────────────────────────────────────────────────────────
def get_agent_pos(blstats):
    return int(blstats[1]), int(blstats[0])   # row, col


def find_chars(glyphs, target_chars, chars=None):
    """用 chars 数组直接匹配字符值，glyphs 保留兼容性但不使用。"""
    positions = []
    if chars is None:
        return positions
    for r in range(chars.shape[0]):
        for c in range(chars.shape[1]):
            if chars[r, c] in target_chars:
                positions.append((r, c))
    return positions


WALKABLE = set(map(ord, ' .@<>_#('))

def bfs_next_action(glyphs, blstats, target_chars, chars=None):
    if chars is None:
        return None
    h, w    = chars.shape
    start   = get_agent_pos(blstats)
    targets = set(find_chars(glyphs, target_chars, chars=chars))
    if not targets:
        return None
    visited = {start}
    queue   = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        if pos in targets:
            if not path:
                return None
            first = path[0]
            dr = max(-1, min(1, first[0] - start[0]))
            dc = max(-1, min(1, first[1] - start[1]))
            return COMPASS.get((dr, dc), WAIT_ACTION)
        r, c = pos
        for ddr, ddc in COMPASS:
            nr, nc = r + ddr, c + ddc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cell = chars[nr, nc]
                if cell in WALKABLE or (nr, nc) in targets:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def adjacent_to_any(blstats, glyphs, target_chars, chars=None):
    """Check if agent is adjacent to any target char."""
    r, c = get_agent_pos(blstats)
    targets = set(find_chars(glyphs, target_chars, chars=chars))
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if (dr, dc) != (0, 0) and (r+dr, c+dc) in targets:
                return True
    return False


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
    name = "GoToStairs"
    def initiate(self, obs): pass
    def act(self, obs):
        a = bfs_next_action(obs["glyphs"], obs["blstats"], {STAIRCASE_CHAR}, chars=obs["chars"])
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        if step >= MAX_OPTION_STEPS:
            return True
        r, c = get_agent_pos(obs["blstats"])
        return (r, c) in set(find_chars(obs["glyphs"], {STAIRCASE_CHAR}, chars=obs["chars"]))


class PickupItemOption(Option):
    name = "PickupItem"
    def initiate(self, obs):
        self._picked = False
    def act(self, obs):
        r, c = get_agent_pos(obs["blstats"])
        items = find_chars(obs["glyphs"], ITEM_CHARS, chars=obs["chars"])
        if (r, c) in items:
            self._picked = True
            return PICKUP_ACTION
        a = bfs_next_action(obs["glyphs"], obs["blstats"], ITEM_CHARS, chars=obs["chars"])
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        if self._picked: return True
        # Immediately terminate if no real items in view
        items = find_chars(obs["glyphs"], ITEM_CHARS, chars=obs["chars"])
        if not items: return True
        if step >= MAX_OPTION_STEPS: return True
        return False


class FindKeyOption(Option):
    """
    KeyRoom-specific: BFS toward '(' key glyph, pick it up.
    This directly addresses the first subtask of KeyRoom.
    """
    name = "FindKey"
    def initiate(self, obs):
        self._picked = False
        self._last_key_pos = None
    def act(self, obs):
        r, c = get_agent_pos(obs["blstats"])
        keys = find_chars(obs["glyphs"], {KEY_CHAR}, chars=obs["chars"])
        if keys:
            self._last_key_pos = keys[0]
        # Trigger pickup if: key visible at agent pos, OR agent is on last known key pos
        on_key = (r, c) in keys or (self._last_key_pos is not None and (r, c) == self._last_key_pos)
        if on_key:
            self._picked = True
            return PICKUP_ACTION
        a = bfs_next_action(obs["glyphs"], obs["blstats"], {KEY_CHAR}, chars=obs["chars"])
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        return step >= MAX_OPTION_STEPS or self._picked


class OpenDoorOption(Option):
    """
    KeyRoom-specific: navigate adjacent to door '+', then apply key.
    Addresses the second subtask of KeyRoom.
    """
    name = "OpenDoor"
    def initiate(self, obs):
        self._opened = False
    def act(self, obs):
        # If already adjacent to door, attempt to apply key
        if adjacent_to_any(obs["blstats"], obs["glyphs"], DOOR_CHARS, chars=obs["chars"]):
            self._opened = True
            return APPLY_ACTION
        # Otherwise BFS toward door
        a = bfs_next_action(obs["glyphs"], obs["blstats"], DOOR_CHARS, chars=obs["chars"])
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))
    def terminate(self, obs, step):
        if self._opened:
            # Verify the door actually disappeared — apply may have failed (no key etc.)
            doors_remain = find_chars(obs["glyphs"], DOOR_CHARS, chars=obs["chars"])
            if not doors_remain:
                return True     # door is genuinely gone
            self._opened = False  # apply failed, keep trying
        if step >= MAX_OPTION_STEPS:
            return True
        # Give the agent enough steps to locate the door before giving up;
        # step > 2 was far too aggressive for rooms larger than 3×3
        doors = find_chars(obs["glyphs"], DOOR_CHARS, chars=obs["chars"])
        if step > MAX_OPTION_STEPS // 4 and not doors:
            return True
        return False


# Registry — order matters for meta-policy action indices
OPTIONS   = [
    ExploreOption(),
    NavigateToStaircaseOption(),
    PickupItemOption(),
    FindKeyOption(),
    OpenDoorOption(),
]
N_OPTIONS = len(OPTIONS)


# ── Option execution ──────────────────────────────────────────────────────────
def execute_option(env, obs, option):
    option.initiate(obs)
    cum_r = 0.0
    step  = 0
    done  = False
    while not option.terminate(obs, step):
        a = option.act(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        done   = terminated or truncated
        cum_r += r
        step  += 1
        if done:
            break
    return obs, cum_r, done, step


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
    def __init__(self, n_options, blstats_size):
        super().__init__()
        self.glyph_enc = GlyphEncoder()
        gh = self.glyph_enc.hidden                    # 256
        self.blstats_enc = nn.Sequential(
            nn.Linear(blstats_size, 64), nn.ReLU(),
        )
        h = gh + 64
        self.policy  = nn.Linear(h, n_options)
        self.value   = nn.Linear(h, 1)
        self.kg_bias = nn.Parameter(torch.zeros(n_options), requires_grad=False)

    def forward(self, g, b):
        f = torch.cat([self.glyph_enc(g), self.blstats_enc(b)], dim=-1)
        return self.policy(f), self.value(f).squeeze(-1)

    def get_action_and_value(self, g, b, action=None):
        logits, val = self(g, b)
        logits = logits + self.kg_bias
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val


# ── Rollout buffer ────────────────────────────────────────────────────────────
class OptionBuffer:
    def __init__(self, n, obs_shape, blstats_size, device):
        self.device    = device
        self.glyphs    = torch.zeros((n, *obs_shape), dtype=torch.long).to(device)
        self.blstats   = torch.zeros((n, blstats_size), dtype=torch.float).to(device)
        self.options   = torch.zeros(n, dtype=torch.long).to(device)
        self.logprobs  = torch.zeros(n).to(device)
        self.rewards   = torch.zeros(n).to(device)
        self.dones     = torch.zeros(n).to(device)
        self.values    = torch.zeros(n).to(device)
        self.durations = torch.zeros(n).to(device)   # option duration k (SMDP)
        self.ptr = 0

    def add(self, g, b, o, lp, r, d, v, k):
        i = self.ptr
        self.glyphs[i]=g; self.blstats[i]=b; self.options[i]=o
        self.logprobs[i]=lp; self.rewards[i]=r; self.dones[i]=d
        self.values[i]=v;  self.durations[i]=k
        self.ptr += 1

    def compute_returns(self, last_v, gamma, lam):
        adv = torch.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.ptr)):
            nv  = last_v if t == self.ptr - 1 else self.values[t + 1]
            nnt = 1.0 - self.dones[t]
            gk  = gamma ** self.durations[t]   # SMDP: γ^k for option of duration k
            delta = self.rewards[t] + gk * nv * nnt - self.values[t]
            gae   = delta + gk * lam * nnt * gae
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
            _, lp, ent, v = model.get_action_and_value(buf.glyphs[i], buf.blstats[i], buf.options[i])
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
    print(f"Options ({N_OPTIONS}): {[o.name for o in OPTIONS]}\n")

    env = gym.make(args.env, observation_keys=("glyphs", "blstats", "chars"),
                   reward_win=1.0, reward_lose=-1.0, penalty_step=-0.001)
    obs_shape    = env.observation_space["glyphs"].shape
    blstats_size = env.observation_space["blstats"].shape[0]

    model     = MetaActorCritic(N_OPTIONS, blstats_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # 加载知识图谱先验 + 子目标概率
    kg_node_probs = {}
    try:
        from kg_planner import load_graph, get_option_weights, dynamic_option_bias, get_node_probs
        graph = load_graph()
        bias = get_option_weights(graph, OPTIONS, args.env)
        model.kg_bias.data = bias.to(device)
        print(f"[KG] Loaded knowledge graph prior: {bias.tolist()}")
        kg_node_probs, _ = get_node_probs(graph, 'agent', 'open')
        if kg_node_probs:
            print(f"[KG] Sub-goal node probs: { {k: f'{v:.4f}' for k, v in kg_node_probs.items()} }")
    except Exception as e:
        print(f"[KG] No prior loaded: {e}")
    base_kg_bias = model.kg_bias.data.clone().cpu()
    buf       = OptionBuffer(ROLLOUT_LEN, obs_shape, blstats_size, device)

    obs, _     = env.reset()
    ep_ret     = 0.0
    ep_count   = 0
    total_prim = 0
    done       = False
    episode_returns = []
    option_counts   = np.zeros(N_OPTIONS, dtype=int)
    start = time.time()
    _prev_chars  = None   # cache for dynamic bias — skip recompute if chars unchanged
    key_obtained = False  # sub-goal flags, reset each episode
    door_opened  = False

    print(f"Training on {args.env} for ~{args.steps} primitive steps...\n")

    while total_prim < args.steps:
        buf.reset()

        for _ in range(ROLLOUT_LEN):
            # ── KG bias: decay magnitude toward zero over training ────────────────
            kg_decay = max(0.0, 1.0 - total_prim / (args.steps * 0.7))
            effective_base = base_kg_bias * kg_decay
            cur_chars = obs.get("chars")
            if cur_chars is not None and not np.array_equal(cur_chars, _prev_chars):
                try:
                    dyn_bias = dynamic_option_bias(graph, obs, OPTIONS, args.env, effective_base, kg_decay)
                    model.kg_bias.data = dyn_bias.to(device)
                except Exception:
                    pass
                _prev_chars = cur_chars.copy()

            # ── Build tensors ─────────────────────────────────────────────────────
            g = torch.tensor(obs[OBS_KEY],    dtype=torch.long,  device=device).unsqueeze(0)
            b = torch.tensor(obs["blstats"],  dtype=torch.float, device=device).unsqueeze(0)
            with torch.no_grad():
                opt_idx, lp, _, v = model.get_action_and_value(g, b)

            option = OPTIONS[opt_idx.item()]
            option_counts[opt_idx.item()] += 1

            # ── Snapshot visible sub-goals before executing the option ────────────
            pre_chars = obs.get("chars")
            key_vis_before  = (pre_chars is not None
                               and np.any(pre_chars == KEY_CHAR)
                               and not key_obtained)
            door_vis_before = (pre_chars is not None
                               and any(np.any(pre_chars == dc) for dc in DOOR_CHARS)
                               and not door_opened)

            obs, cum_r, done, prim_n = execute_option(env, obs, option)

            # ── Intrinsic sub-goal rewards: R = η · P_node · decay ───────────────
            # P_node from KG; decay matches kg_bias schedule (zero at 70% of training)
            post_chars = obs.get("chars")
            if key_vis_before and post_chars is not None and not np.any(post_chars == KEY_CHAR):
                cum_r       += ETA_REWARD * kg_node_probs.get('key', 0.0) * kg_decay
                key_obtained = True
            if door_vis_before and post_chars is not None and not any(np.any(post_chars == dc) for dc in DOOR_CHARS):
                cum_r       += ETA_REWARD * kg_node_probs.get('door', 0.0) * kg_decay
                door_opened = True

            ep_ret     += cum_r
            total_prim += prim_n

            buf.add(g.squeeze(0), b.squeeze(0), opt_idx, lp, cum_r, float(done), v, float(prim_n))

            if done:
                ep_count += 1
                episode_returns.append(ep_ret)
                if ep_count % 10 == 0:
                    mean_ret = np.mean(episode_returns[-10:])
                    sps      = total_prim / (time.time() - start)
                    counts   = " | ".join(
                        f"{OPTIONS[i].name}:{option_counts[i]}" for i in range(N_OPTIONS)
                    )
                    print(f"PrimSteps:{total_prim:7d} | Eps:{ep_count:5d} | "
                          f"MeanRet:{mean_ret:7.3f} | SPS:{sps:.0f}")
                    print(f"  [{counts}]")
                ep_ret = 0.0
                obs, _ = env.reset()
                done = False
                _prev_chars  = None   # force bias refresh at episode start
                key_obtained = False  # reset sub-goal flags for new episode
                door_opened  = False

            if total_prim >= args.steps:
                break

        g_last = torch.tensor(obs[OBS_KEY],   dtype=torch.long,  device=device).unsqueeze(0)
        b_last = torch.tensor(obs["blstats"], dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, lv = model.get_action_and_value(g_last, b_last)
        ppo_update(model, optimizer, buf, lv.item() if not done else 0.0, device)

    env.close()
    final = np.mean(episode_returns[-20:]) if episode_returns else 0.0
    print(f"\nTraining complete. Final mean return (last 20): {final:.3f}")
    print("\nOption usage:")
    for i, o in enumerate(OPTIONS):
        print(f"  {o.name:25s}: {option_counts[i]:6d} times")
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",       default="MiniHack-KeyRoom-S5-v0")
    parser.add_argument("--steps",     type=int, default=200_000)
    parser.add_argument("--save_path", default="options_ppo.pt")
    train(parser.parse_args())
