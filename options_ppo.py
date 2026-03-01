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
import os
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
GAMMA              = 0.99
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
ENTROPY_COEF_START = 0.05    # high entropy early → explore options freely
ENTROPY_COEF_END   = 0.005   # low entropy late  → commit to learned policy
VALUE_COEF         = 0.5
MAX_GRAD           = 0.5
LR                 = 2.5e-4
ROLLOUT_LEN        = 128     # larger buffer → more stable gradient estimates
N_EPOCHS           = 6       # more SGD passes per rollout
BATCH_SIZE         = 32
MAX_OPTION_STEPS   = 30      # shorter cap prevents long futile loops
OBS_KEY          = "glyphs"

# Intrinsic reward scale; per-sub-goal magnitude = ETA * P_node * decay
# P_node comes from kg_planner.get_node_probs; decay matches kg_bias schedule
ETA_REWARD = 0.1

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
    if chars is None:
        return []
    return [tuple(p) for p in np.argwhere(np.isin(chars, list(target_chars)))]


WALKABLE = set(map(ord, ' .@<>_#('))

def bfs_next_action(glyphs, blstats, target_chars, chars=None):
    if chars is None:
        return None
    h, w    = chars.shape
    start   = get_agent_pos(blstats)
    targets = set(find_chars(glyphs, target_chars, chars=chars))
    if not targets or start in targets:
        return None
    visited = {start}
    parent  = {start: None}   # pos → parent pos；用父指针代替存整条路径
    queue   = deque([start])
    while queue:
        pos = queue.popleft()
        if pos in targets:
            # 反向追溯到起点的第一步
            while parent[pos] != start:
                pos = parent[pos]
            dr = max(-1, min(1, pos[0] - start[0]))
            dc = max(-1, min(1, pos[1] - start[1]))
            return COMPASS.get((dr, dc), WAIT_ACTION)
        r, c = pos
        for ddr, ddc in COMPASS:
            nr, nc = r + ddr, c + ddc
            npos = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and npos not in visited:
                if chars[nr, nc] in WALKABLE or npos in targets:
                    visited.add(npos)
                    parent[npos] = pos
                    queue.append(npos)
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
    """优先走未访问过的相邻格，退化为随机游走当所有邻居都已访问。"""
    name = "Explore"

    def initiate(self, obs):
        self._visited = set()
        self._visited.add(get_agent_pos(obs["blstats"]))

    def act(self, obs):
        r, c  = get_agent_pos(obs["blstats"])
        self._visited.add((r, c))
        chars = obs["chars"]
        h, w  = chars.shape
        unvisited = [
            action for (dr, dc), action in COMPASS.items()
            if 0 <= r+dr < h and 0 <= c+dc < w
            and chars[r+dr, c+dc] in WALKABLE
            and (r+dr, c+dc) not in self._visited
        ]
        if unvisited:
            return int(np.random.choice(unvisited))
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
        # 踩上 '>' 会触发 done=True（episode 结束），execute_option 会直接 break；
        # 这里仅作超时兜底
        return step >= MAX_OPTION_STEPS


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
    KeyRoom-specific: navigate adjacent to door '+', then unlock+open with key.

    NetHack 开门分两个独立动作：
      unlock:  APPLY ('a') → 选 key ('a') → 朝门方向 → 门解锁，但仍显示 '+'
      open:    走进已解锁的门（bump）→ '+' 变成 '-'/'|' 并消失 → 检测成功

    完整四步状态机（每个 cycle）：
      navigate → apply1 → apply2 → unlock → (back to navigate)
        apply1: APPLY_ACTION  — 进入 apply 模式，"Apply which item?"
        apply2: APPLY_ACTION  — 选 slot 'a'（key 固定在此 slot），"In what direction?"
        unlock: cached_dir    — 回答方向 → 门解锁（仍显示 '+'）
      navigate 下一步:
        BFS 相邻门 → 发 cached_dir 走进已解锁门 → 门消失（成功）

    关键设计：
    - 到达门旁时缓存方向（APPLY 后 menu 覆盖地图，所以提前缓存）
    - terminate() 在 apply1/apply2/unlock 三步中跳过门检查（避免 menu overlay 误判）
    - message 包含 "apply" 或 "direction" 关键词时额外保护，防止误终止
    - _cycle_count 统计完整 cycle，达到阈值则放弃
    """
    name = "OpenDoor"

    # NLE message 中出现这些词说明正处于交互菜单，不应判定失败
    _MENU_KEYWORDS = (b"apply", b"direction", b"what do you want", b"which item")

    def initiate(self, obs):
        # 状态机: navigate → apply1 → apply2 → unlock → navigate → ...
        self._step        = 'navigate'
        self._cached_dir  = None
        self._cycle_count = 0

    @staticmethod
    def _has_key(obs):
        """通过 inv_letters 观测判断 slot 'a' 是否有物品（即钥匙是否在背包里）。
        inv_letters 是 NLE 原生观测，每个元素是对应 slot 字母的 ASCII 码，0 表示空。
        同时解决了 slot 假设问题：如果 slot 'a' 真的没东西，就不发 APPLY，
        避免在错误 slot 上浪费动作。
        """
        letters = obs.get("inv_letters")
        if letters is None:
            return True   # 无法观测时放行，保持之前行为
        return bool(letters[0])   # slot 'a' 非空 → 有物品可 apply

    @staticmethod
    def _in_menu(obs):
        """检查 message 是否包含交互菜单提示。"""
        msg = obs.get("message")
        if msg is None:
            return False
        raw = bytes(msg.flatten()).lower()
        return any(kw in raw for kw in OpenDoorOption._MENU_KEYWORDS)

    def act(self, obs):
        if self._step == 'apply1':
            self._step = 'apply2'
            return APPLY_ACTION                   # 选 inventory slot 'a'（key）

        if self._step == 'apply2':
            self._step = 'unlock'
            return self._cached_dir if self._cached_dir is not None else WAIT_ACTION
            # 回答 "In what direction?" → 门解锁，仍显示 '+'

        if self._step == 'unlock':
            # 走进已解锁的门 → '+' 消失
            self._step = 'navigate'
            self._cycle_count += 1
            return self._cached_dir if self._cached_dir is not None else WAIT_ACTION

        # navigate 阶段：BFS 向门移动；相邻后启动四步序列
        if adjacent_to_any(obs["blstats"], obs["glyphs"], DOOR_CHARS, chars=obs["chars"]):
            self._cached_dir = bfs_next_action(
                obs["glyphs"], obs["blstats"], DOOR_CHARS, chars=obs["chars"]
            )
            self._step = 'apply1'
            return APPLY_ACTION                   # 进入 apply 模式

        a = bfs_next_action(obs["glyphs"], obs["blstats"], DOOR_CHARS, chars=obs["chars"])
        return a if a is not None else int(np.random.choice(list(COMPASS.values())))

    def terminate(self, obs, step):
        # 背包 slot 'a' 为空 → 钥匙还没拿到，立刻退出让 meta-policy 重新决策
        if not self._has_key(obs):
            return True
        # 序列进行中或处于交互菜单：跳过门存在检查
        if self._step in ('apply1', 'apply2', 'unlock') or self._in_menu(obs):
            return step >= MAX_OPTION_STEPS

        doors_remain = find_chars(obs["glyphs"], DOOR_CHARS, chars=obs["chars"])
        if not doors_remain:
            return True             # 门消失 → 开门成功
        if self._cycle_count >= 3:
            return True             # 三次 cycle 均失败，放弃
        if step >= MAX_OPTION_STEPS:
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
N_OPTIONS     = len(OPTIONS)
OPEN_DOOR_IDX = next(i for i, o in enumerate(OPTIONS) if o.name == "OpenDoor")


# ── Option execution ──────────────────────────────────────────────────────────
def execute_option(env, obs, option):
    """执行一个 option 直到其终止条件满足。

    SMDP 要求：每次 option 调用必须至少消耗 1 个 primitive step。
    因此使用 do-while 结构（先 act 再 terminate），杜绝"零步选项死循环"：
    即 terminate() 在 step=0 时立刻返回 True → meta-policy 拿回控制权
    → 同一帧画面被反复调用 → option 计数暴涨、SPS 断崖下跌的问题。
    """
    option.initiate(obs)
    cum_r = 0.0
    step  = 0
    done  = False
    while True:
        a = option.act(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        done   = terminated or truncated
        cum_r += r
        step  += 1
        if done or option.terminate(obs, step):
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

    def get_action_and_value(self, g, b, action=None, mask=None):
        logits, val = self(g, b)
        logits = logits + self.kg_bias
        if mask is not None:
            # 将无效 option 的 logit 置为 -inf，使其概率为 0
            logits = logits.masked_fill(~mask, float('-inf'))
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
        # 动作掩码：记录每步选 option 时的合法集合，PPO update 时复用保证 ratio 一致
        self.masks     = torch.ones((n, N_OPTIONS), dtype=torch.bool).to(device)
        self.ptr = 0

    def add(self, g, b, o, lp, r, d, v, k, mask):
        i = self.ptr
        self.glyphs[i]=g; self.blstats[i]=b; self.options[i]=o
        self.logprobs[i]=lp; self.rewards[i]=r; self.dones[i]=d
        self.values[i]=v;  self.durations[i]=k; self.masks[i]=mask
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
def ppo_update(model, optimizer, buf, last_v, device, entropy_coef=ENTROPY_COEF_END):
    rets, adv = buf.compute_returns(last_v, GAMMA, GAE_LAMBDA)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(N_EPOCHS):
        idx = np.random.permutation(buf.ptr)
        for s in range(0, buf.ptr, BATCH_SIZE):
            i = torch.tensor(idx[s:s + BATCH_SIZE], device=device)
            _, lp, ent, v = model.get_action_and_value(buf.glyphs[i], buf.blstats[i], buf.options[i], mask=buf.masks[i])
            r    = (lp - buf.logprobs[i]).exp()
            loss = (
                -torch.min(r * adv[i],
                           torch.clamp(r, 1-CLIP_EPS, 1+CLIP_EPS) * adv[i]).mean()
                + VALUE_COEF * 0.5 * (v - rets[i]).pow(2).mean()
                - entropy_coef * ent.mean()
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

    env = gym.make(args.env, observation_keys=("glyphs", "blstats", "chars", "message", "inv_letters"),
                   reward_win=args.reward_win, reward_lose=-1.0, penalty_step=-0.001)
    print(f"Reward on win: {args.reward_win}")
    obs_shape    = env.observation_space["glyphs"].shape
    blstats_size = env.observation_space["blstats"].shape[0]

    model     = MetaActorCritic(N_OPTIONS, blstats_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # ── 加载知识图谱，构建 KGPathState ──────────────────────────────────────────
    kg_path_state = None
    try:
        from kg_planner import load_graph, make_kg_path_state
        graph = load_graph()
        kg_path_state, init_bias = make_kg_path_state(
            graph, OPTIONS, args.env, eta=ETA_REWARD
        )
        if kg_path_state is not None:
            model.kg_bias.data = init_bias.to(device)
    except Exception as e:
        print(f"[KG] No prior loaded: {e}")

    buf = OptionBuffer(ROLLOUT_LEN, obs_shape, blstats_size, device)

    obs, _     = env.reset()
    ep_ret     = 0.0
    ep_count   = 0
    total_prim = 0
    done       = False
    episode_returns = []
    option_counts   = np.zeros(N_OPTIONS, dtype=int)
    start = time.time()

    # KGPathState: 每个 episode 重置
    if kg_path_state is not None:
        kg_path_state.reset(obs)

    print(f"Training on {args.env} for ~{args.steps} primitive steps...\n")

    while total_prim < args.steps:
        buf.reset()

        for _ in range(ROLLOUT_LEN):
            # ── Entropy decay: high early (explore), low late (exploit) ──────────
            progress      = total_prim / args.steps
            entropy_coef  = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress

            # ── KG bias decay: floor at 0.15 so KG always has residual influence ─
            kg_decay = max(0.15, 1.0 - total_prim / (args.steps * 0.7))

            # ── 动态 KG bias：来自当前路径位置（Dynamic Rerooting）───────────────
            if kg_path_state is not None:
                dyn_bias = kg_path_state.get_bias(kg_decay)
                model.kg_bias.data = dyn_bias.to(device)

            # ── Build tensors ──────────────────────────────────────────────────────
            g = torch.tensor(obs[OBS_KEY],   dtype=torch.long,  device=device).unsqueeze(0)
            b = torch.tensor(obs["blstats"], dtype=torch.float, device=device).unsqueeze(0)

            # ── 动作掩码：没有钥匙时禁止选 OpenDoor ───────────────────────────────
            inv = obs.get("inv_letters")
            has_key = bool(inv[0]) if inv is not None else True
            opt_mask = torch.ones(N_OPTIONS, dtype=torch.bool, device=device)
            if not has_key:
                opt_mask[OPEN_DOOR_IDX] = False

            with torch.no_grad():
                opt_idx, lp, _, v = model.get_action_and_value(g, b, mask=opt_mask.unsqueeze(0))

            option = OPTIONS[opt_idx.item()]
            option_counts[opt_idx.item()] += 1

            # ── 执行 option，记录执行前观测 ───────────────────────────────────────
            # NLE observation arrays are views into shared C buffers; copy so
            # that subsequent env.step() calls don't overwrite pre_obs in-place.
            pre_obs = {k: v.copy() if isinstance(v, np.ndarray) else v
                       for k, v in obs.items()}
            obs, cum_r, done, prim_n = execute_option(env, obs, option)

            # ── KGPathState: 先记录 option 尝试（学习顺序依赖），再检测边穿越 ────
            # record_option_attempt 在 update() 推进路径步骤之前调用，
            # 这样它能知道"此时路径在哪一步"，从而判断是否是超前调用
            if kg_path_state is not None:
                kg_path_state.record_option_attempt(option.name, pre_obs, obs)
                intr_r  = kg_path_state.update(pre_obs, obs, option_name=option.name, kg_decay=kg_decay)
                cum_r  += intr_r

            ep_ret     += cum_r
            total_prim += prim_n

            buf.add(g.squeeze(0), b.squeeze(0), opt_idx, lp, cum_r, float(done), v, float(prim_n), opt_mask)

            if done:
                ep_count += 1
                episode_returns.append(ep_ret)
                success = ep_ret >= args.reward_win * 0.5   # only genuine wins (≥ 2.5 with reward_win=5.0)
                if kg_path_state is not None:
                    if success:
                        # 发现模式：同时传入 pre_obs（楼梯未被 @ 遮挡时可见）和 terminal obs
                        kg_path_state.discover_from_victory(obs, pre_victory_obs=pre_obs)
                    kg_path_state.end_episode(success)   # EMA 自更新 node_probs
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
                done   = False
                if kg_path_state is not None:
                    kg_path_state.reset(obs)   # 每 episode 重置路径状态

            if total_prim >= args.steps:
                break

        g_last = torch.tensor(obs[OBS_KEY],   dtype=torch.long,  device=device).unsqueeze(0)
        b_last = torch.tensor(obs["blstats"], dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, lv = model.get_action_and_value(g_last, b_last)
        ppo_update(model, optimizer, buf, lv.item() if not done else 0.0, device, entropy_coef)

    env.close()
    final = np.mean(episode_returns[-20:]) if episode_returns else 0.0
    print(f"\nTraining complete. Final mean return (last 20): {final:.3f}")
    print("\nOption usage:")
    for i, o in enumerate(OPTIONS):
        print(f"  {o.name:25s}: {option_counts[i]:6d} times")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")
    if kg_path_state is not None:
        kg_path_state.save_updated_probs()
        print("[KG] Saved learned node probabilities to data/kg_learned_probs.json")
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="MiniHack-KeyRoom-S5-v0")
    parser.add_argument("--steps",      type=int,   default=200_000)
    parser.add_argument("--save_path",  default="options_ppo.pt")
    parser.add_argument("--reward-win", dest="reward_win", type=float, default=5.0,
                        help="Reward given when task is solved (default: 5.0)")
    train(parser.parse_args())
