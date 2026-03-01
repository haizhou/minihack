"""
kg_planner.py — Knowledge-graph path planner + dynamic state tracker
=====================================================================
KGPathState: 核心改进
  - 每个 episode 独立跟踪 agent 在 KG 路径上的当前位置
  - bias 向量随路径推进动态缩减（只剩余路径节点获得 boost）
  - intrinsic reward = KG 边穿越时目标节点的概率（来自 world knowledge，非人工设置）
  - 边穿越检测: 基于 edge relation 语义 + NODE_TO_CHAR glyph 映射
"""
import json
import os
import numpy as np
import torch
from collections import deque


# ── KG 节点 → 对应 glyph 字符集（连接符号 KG 与视觉观测的桥梁）──────────────────
NODE_TO_CHAR = {
    'key':    {ord('(')},
    'door':   {ord('+')},
    'open':   {ord('.')},
    'stairs': {ord('>')},
}

# Option 关键词 → 负责到达的 KG 节点（option 命名约定的语义映射）
OPTION_TO_NODE = {
    'findkey':    'key',
    'find':       'key',
    'pickup':     'key',
    'opendoor':   'door',
    'open':       'door',
    'gotostairs': 'stairs',
    'stair':      'stairs',
}

# KG edge relation → 穿越该边时使用的观测检测方式
# 语义来自 edge relation 本身的含义，不依赖特定环境规则：
#   can_pickup:   物品从地图消失（被拾取）→ glyph 数量减少
#   enables:      持有物品后触发状态变化 → 目标 glyph 数量减少
#   state_change: 实体状态改变         → 目标 glyph 数量减少
EDGE_DETECTION = {
    'can_pickup':   'glyph_decrease',
    'enables':      'glyph_decrease',
    'state_change': 'glyph_decrease',
}


def load_graph(path='data/knowledge_graph.json'):
    with open(path) as f:
        return json.load(f)['graph']


def bfs_full_path(graph, start, goal):
    queue   = deque([(start, [])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for e in graph.get(node, []):
            if e['target'] not in visited:
                visited.add(e['target'])
                queue.append((e['target'], path + [(node, e['relation'], e['target'])]))
    return []


def get_node_probs(graph, start, goal, n_other_estimate=138):
    path = bfs_full_path(graph, start, goal)
    if not path:
        return {}, []
    path_nodes  = [t for s, r, t in path if t != goal]
    n_path      = len(path_nodes)
    path_scores = torch.tensor([1.0 / (i + 1) for i in range(n_path)])
    path_probs  = path_scores / path_scores.sum() * 0.9
    probs = {node: p for node, p in zip(path_nodes, path_probs.tolist())}
    return probs, path


def _bias_for_path_segment(remaining_path, node_probs, options, bias_scale=3.0, path_decay=0.6):
    """
    给定剩余路径片段，计算 option bias 向量。
    只有剩余路径中的目标节点对应的 option 获得正向 bias，
    已完成节点自动退出，不需要任何手动屏蔽。
    """
    n      = len(options)
    bias   = torch.zeros(n)
    seen   = set()

    for step_idx, (src, rel, dst) in enumerate(remaining_path):
        if dst == 'open':        # 终点不对应具体 option
            continue
        decay    = path_decay ** step_idx
        node_p   = node_probs.get(dst, 0.0)
        for i, opt in enumerate(options):
            name = opt.name.lower()
            matched_node = next(
                (nd for kw, nd in OPTION_TO_NODE.items() if kw in name and nd == dst),
                None
            )
            if matched_node and (i, dst) not in seen:
                bias[i] += node_p * bias_scale * decay
                seen.add((i, dst))

    for i, opt in enumerate(options):
        if 'explore' in opt.name.lower():
            bias[i] -= 0.5

    return bias


def get_option_weights(graph, options, env_name, start='agent', goal='open'):
    """初始静态权重：训练开始前用于打印，实际 bias 由 KGPathState 动态管理。"""
    if 'KeyRoom' not in env_name:
        return torch.zeros(len(options))

    node_probs, path = get_node_probs(graph, start, goal)

    print(f"[KG] Optimal path:")
    for s, r, t in path:
        print(f"[KG]   {s:20s} -[{r}]-> {t}")
    print(f"[KG] Node probabilities:")
    for node, p in node_probs.items():
        print(f"[KG]   {node:20s}: p={p:.4f}")

    weights = _bias_for_path_segment(path, node_probs, options)
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# KGPathState — 核心新增：每个 episode 的 KG 路径状态追踪器
# ══════════════════════════════════════════════════════════════════════════════
class KGPathState:
    """
    动态跟踪 agent 在 KG 路径上的当前位置，并据此提供：
      1. 动态 option bias 向量（仅剩余子目标获得 boost）
      2. KG-derived intrinsic reward（边穿越时释放，大小 = 目标节点概率 * eta）

    边穿越检测策略（来自 edge relation 语义，不依赖环境特定规则）：
      - 'glyph_decrease': 目标节点对应的 glyph 字符在地图上的数量减少
        （覆盖 can_pickup / enables / state_change 三种 relation）

    所有概率值来自 world_knowledge 学习到的 KG，eta 是唯一超参数。
    """

    def __init__(self, path, node_probs, options, eta=0.5, self_update_lr=0.01):
        """
        path:            [(src, relation, dst), ...] from bfs_full_path
        node_probs:      {node: float} from get_node_probs  (learned from world knowledge)
        options:         list of Option objects
        eta:             intrinsic reward scaling factor
        self_update_lr:  EMA learning rate for self-updating node probs from experience
        """
        self.path            = path
        self.node_probs      = dict(node_probs)   # mutable copy for self-update
        self.options         = options
        self.eta             = eta
        self.self_update_lr  = self_update_lr

        # 预计算每个路径位置对应的 bias 向量（位置 i = 前 i 条边已完成）
        self._bias_cache = [
            _bias_for_path_segment(path[i:], node_probs, options)
            for i in range(len(path) + 1)
        ]

        self.current_step = 0   # 当前在 path 中的索引
        self._prev_chars  = None
        self._last_key_obs_pos = None   # 本 episode 最后一次看到 key 的地图坐标（遮挡修复用）

        # episode 内曾出现过的节点集合（用于 discover：避免被 @ 遮挡导致扫描失败）
        self._nodes_seen_this_episode: set = set()

        # ── Self-update state ──────────────────────────────────────────────────
        self._episode_traversed = []   # edge indices traversed this episode
        self._total_episodes    = 0
        self._update_interval   = 20   # recompute bias cache every N episodes

        # ── Ordering constraint learning ───────────────────────────────────────
        # Agent discovers dependencies by trying options prematurely and failing.
        # _premature_calls[node] = how many times that option was called before
        #   its prerequisite sub-goal was reached.
        # _ordering_constraints[node] = prerequisite node that must come first.
        # _max_premature: threshold of failures before constraint is registered.
        self._premature_calls       = {}   # {node: int}
        self._ordering_constraints  = {}   # {node: prerequisite_node}
        self._max_premature         = 15   # fast enough to learn, stable enough to trust

    def reset(self, init_obs):
        """每个 episode 开始时重置状态。"""
        self.current_step = 0
        self._prev_chars  = init_obs.get('chars')
        self._last_key_obs_pos = None
        self._nodes_seen_this_episode = set()
        # 扫描初始观测：记录初始可见节点 + 定位 key 初始坐标
        chars = init_obs.get('chars')
        if chars is not None:
            key_char_set = NODE_TO_CHAR.get('key', set())
            for rr in range(chars.shape[0]):
                for cc in range(chars.shape[1]):
                    if chars[rr, cc] in key_char_set:
                        self._last_key_obs_pos = (rr, cc)
                        break
            for node_name, char_set in NODE_TO_CHAR.items():
                if any(int(np.sum(chars == c)) > 0 for c in char_set):
                    self._nodes_seen_this_episode.add(node_name)

    def update(self, pre_obs, post_obs, option_name=''):
        """
        option 执行完后调用。比较执行前后的观测，判断是否穿越了下一条 KG 边。

        返回: intrinsic_reward (float)
          - 穿越了边: = node_probs[dst] * eta
          - 未穿越:   = 0.0
        """
        # 持续记录 episode 内见过的所有节点（用于 discover_from_victory 翻旧账）
        pre_chars_scan = pre_obs.get('chars')
        if pre_chars_scan is not None:
            for _node, _cset in NODE_TO_CHAR.items():
                if any(int(np.sum(pre_chars_scan == c)) > 0 for c in _cset):
                    self._nodes_seen_this_episode.add(_node)

        if self.current_step >= len(self.path):
            self._prev_chars = post_obs.get('chars')
            return 0.0

        src, rel, dst = self.path[self.current_step]
        detection     = EDGE_DETECTION.get(rel, 'glyph_decrease')
        traversed     = False

        # 更新 key 最后可见坐标（在做检测前先记录，以便遮挡兜底使用）
        if dst == 'key':
            pre_chars_arr = pre_obs.get('chars')
            if pre_chars_arr is not None:
                key_char_set = NODE_TO_CHAR.get('key', set())
                for rr in range(pre_chars_arr.shape[0]):
                    for cc in range(pre_chars_arr.shape[1]):
                        if pre_chars_arr[rr, cc] in key_char_set:
                            self._last_key_obs_pos = (rr, cc)
                            break

        if detection == 'glyph_decrease':
            traversed = self._glyph_decreased(pre_obs, post_obs, dst)

        # 遮挡兜底：agent 站在 key 上时 pre_count/post_count 均为 0，导致 glyph_decrease 漏判
        # 条件：1) 主检测未通过  2) 目标节点是 key  3) 执行的是拾取类 option
        #       4) 本 episode 曾见过 key（_last_key_obs_pos 不为 None）
        #       5) option 结束后 key 已消失
        if not traversed and dst == 'key' and self._last_key_obs_pos is not None:
            is_pickup = any(kw in option_name.lower() for kw in ('findkey', 'pickup'))
            if is_pickup:
                pre_blstats = pre_obs.get('blstats')
                if pre_blstats is not None:
                    agent_pos = (int(pre_blstats[1]), int(pre_blstats[0]))
                    if agent_pos == self._last_key_obs_pos:
                        post_chars_arr = post_obs.get('chars')
                        if post_chars_arr is not None:
                            post_key_count = sum(
                                int(np.sum(post_chars_arr == c))
                                for c in NODE_TO_CHAR.get('key', set())
                            )
                            if post_key_count == 0:
                                traversed = True

        intr_r = 0.0
        if traversed:
            self._episode_traversed.append(self.current_step)   # record before increment
            self.current_step += 1
            intr_r = self.node_probs.get(dst, 0.0) * self.eta
            if intr_r > 0:
                print(f"  [KG✓] Edge traversed: {src} -[{rel}]-> {dst}  "
                      f"| step={self.current_step}/{len(self.path)}"
                      f"  intr_r={intr_r:.4f}")

        self._prev_chars = post_obs.get('chars')
        return intr_r

    def get_bias(self, kg_decay=1.0):
        """
        返回当前路径位置的动态 bias 向量，乘以 kg_decay。
        已完成的子目标节点不再出现在 bias 中。
        叠加已学到的顺序约束惩罚（prerequisite 未完成时压制被依赖 option）。
        """
        idx       = min(self.current_step, len(self._bias_cache) - 1)
        base_bias = self._bias_cache[idx] * kg_decay

        # Apply learned ordering constraints
        if self._ordering_constraints and kg_decay > 0:
            penalty = torch.zeros_like(base_bias)
            for i, opt in enumerate(self.options):
                opt_node = next(
                    (nd for kw, nd in OPTION_TO_NODE.items() if kw in opt.name.lower()),
                    None,
                )
                if opt_node not in self._ordering_constraints:
                    continue
                prereq_node = self._ordering_constraints[opt_node]
                # Find the path step at which prereq_node is reached
                prereq_step = next(
                    (j for j, (_, _, t) in enumerate(self.path) if t == prereq_node),
                    None,
                )
                if prereq_step is not None and self.current_step <= prereq_step:
                    # Prerequisite not yet achieved → penalise this option
                    penalty[i] -= 2.5
            base_bias = base_bias + penalty * kg_decay

        return base_bias

    def progress(self):
        """返回 (current_step, total_steps) 用于日志。"""
        return self.current_step, len(self.path)

    def record_option_attempt(self, option_name: str, pre_obs, post_obs):
        """
        每次 option 执行完、update() 推进路径步骤之前调用。

        检测"超前调用"：该 option 对应的 KG 节点不是当前下一步，而是未来某步。
        每次超前调用计数 +1；达到 _max_premature 阈值后，
        自动注册 ordering constraint（该节点需要当前下一步先完成）。

        这是 agent 无需外部指导自行发现 option 依赖关系的机制。
        """
        opt_node = next(
            (nd for kw, nd in OPTION_TO_NODE.items() if kw in option_name.lower()),
            None,
        )
        if opt_node is None or self.current_step >= len(self.path):
            return

        _, _, current_next = self.path[self.current_step]
        if opt_node == current_next:
            return  # 正确顺序，不记录

        # 是否是路径中靠后的节点？
        future_nodes = [t for _, _, t in self.path[self.current_step + 1:]]
        if opt_node not in future_nodes:
            return  # 不在路径中，忽略

        # 超前调用：计数
        self._premature_calls[opt_node] = self._premature_calls.get(opt_node, 0) + 1
        count = self._premature_calls[opt_node]

        if count == self._max_premature:
            # 阈值触发：注册 ordering constraint
            self._ordering_constraints[opt_node] = current_next
            print(
                f"  [KG🧠] Learned dependency: '{opt_node}' requires '{current_next}' first"
                f"  (after {count} premature attempts)"
            )
            # get_bias() 会在下次调用时自动应用此约束，无需重算 cache

    def discover_from_victory(self, victory_obs, pre_victory_obs=None):
        """
        发现模式：agent 胜利时调用。

        扫描胜利状态的 chars，找出 NODE_TO_CHAR 里有对应 glyph 但
        当前不在 KG 路径中的节点。将其作为新节点追加到路径末尾，
        并以 'discovered' relation 连接到前一个子目标。

        注意：agent 胜利时通常站在目标节点（楼梯 '>'）上，
        此时 chars 显示 '@' 而非 '>'，会导致扫描失败。
        因此同时扫描 pre_victory_obs（胜利 option 开始前的观测），
        彼时 agent 尚未站上目标节点，glyph 可见。
        """
        # 从三个来源合并候选节点，跨候选去重：
        #   1. episode 内任意时刻见过的节点（主要来源，翻旧账）
        #   2. pre_victory_obs：winning option 开始前一帧（agent 未站上目标节点）
        #   3. victory_obs：terminal 帧（目标节点可能被 @ 遮挡，作为兜底）
        found_nodes: dict = {n: True for n in self._nodes_seen_this_episode}

        path_nodes = {t for _, _, t in self.path}
        last_node  = self.path[-1][2] if self.path else 'agent'

        for obs in filter(None, [pre_victory_obs, victory_obs]):
            chars = obs.get('chars')
            if chars is None:
                continue
            for node_name, char_set in NODE_TO_CHAR.items():
                if node_name not in found_nodes:
                    if any(int(np.sum(chars == c)) > 0 for c in char_set):
                        found_nodes[node_name] = True

        added = False
        for node_name in found_nodes:
            if node_name in path_nodes:
                continue   # 已在路径中，跳过
            new_edge = (last_node, 'discovered', node_name)
            self.path.append(new_edge)
            self.node_probs[node_name] = 0.5   # 初始中性概率，EMA 会自动调整
            last_node = node_name              # 支持链式发现
            path_nodes.add(node_name)
            added = True
            print(
                f"  [KG🔍] Discovered: {new_edge[0]} -[discovered]-> {node_name}"
                f"  (glyph visible in pre-victory observation)"
            )

        if added:
            # 扩充路径后重建 bias cache，新节点纳入规划
            self._bias_cache = [
                _bias_for_path_segment(self.path[i:], self.node_probs, self.options)
                for i in range(len(self.path) + 1)
            ]

    def end_episode(self, success: bool):
        """
        每个 episode 结束时调用。
        用本轮经验对 node_probs 做 EMA 自更新，无需人工指导。

        更新规则（EMA）:
          target = 1.0  → 该边被穿越 且 episode 成功（正向强化）
          target = 0.0  → 其他情况（温和压制）

        含义：每条边的 prob 收敛到「穿越该边 + 成功」的经验频率。
        """
        # 胜利回溯：成功时将所有未穿越路径步骤标记为已穿越（确保全路径获得正向 EMA 更新）
        if success:
            for i in range(len(self.path)):
                if i not in self._episode_traversed:
                    self._episode_traversed.append(i)

        lr = self.self_update_lr
        for i, (src, rel, dst) in enumerate(self.path):
            if dst not in self.node_probs:
                continue
            traversed = i in self._episode_traversed
            target    = 1.0 if (traversed and success) else 0.0
            old_p     = self.node_probs[dst]
            new_p     = (1 - lr) * old_p + lr * target
            self.node_probs[dst] = float(np.clip(new_p, 0.15, 0.99))

        self._episode_traversed = []
        self.current_step       = 0
        self._total_episodes   += 1

        # Recompute bias cache periodically so option weights reflect learned probs
        if self._total_episodes % self._update_interval == 0:
            self._recompute_bias_cache()
            prob_str = ", ".join(f"{k}:{v:.3f}" for k, v in self.node_probs.items())
            print(f"  [KG✦] Self-updated (ep {self._total_episodes}): {prob_str}")

    def _recompute_bias_cache(self):
        """Rebuild bias vectors from current (self-updated) node_probs."""
        self._bias_cache = [
            _bias_for_path_segment(self.path[i:], self.node_probs, self.options)
            for i in range(len(self.path) + 1)
        ]

    def save_updated_probs(self, path='data/kg_learned_probs.json'):
        """Persist learned node probabilities for inspection / warm-start."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'node_probs':      self.node_probs,
                'total_episodes':  self._total_episodes,
            }, f, indent=2)
        print(f"  [KG] Saved learned probs → {path}")

    # ── 内部：glyph 数量减少检测 ───────────────────────────────────────────────
    def _glyph_decreased(self, pre_obs, post_obs, dst_node):
        """
        检查 dst_node 对应的 glyph 字符在地图上的数量是否减少。
        使用 NODE_TO_CHAR 映射（KG 节点 → 环境 glyph，两者构建时一致）。
        """
        target_chars = NODE_TO_CHAR.get(dst_node)
        if not target_chars:
            return False

        pre_chars  = pre_obs.get('chars')
        post_chars = post_obs.get('chars')
        if pre_chars is None or post_chars is None:
            return False

        pre_count  = sum(int(np.sum(pre_chars  == c)) for c in target_chars)
        post_count = sum(int(np.sum(post_chars == c)) for c in target_chars)
        return post_count < pre_count


# ── 兼容旧接口（训练脚本初始化时调用） ──────────────────────────────────────────
def make_kg_path_state(graph, options, env_name, eta=0.5, start='agent', goal='open'):
    """
    工厂函数：从已加载的 graph 构建 KGPathState。
    返回 (kg_path_state, initial_bias_tensor)。
    如果不是 KeyRoom 环境，返回 (None, zeros)。
    """
    if 'KeyRoom' not in env_name:
        return None, torch.zeros(len(options))

    node_probs, path = get_node_probs(graph, start, goal)
    if not path:
        return None, torch.zeros(len(options))

    state         = KGPathState(path, node_probs, options, eta=eta)
    initial_bias  = state.get_bias(kg_decay=1.0)
    print(f"[KG] Loaded knowledge graph prior: {initial_bias.tolist()}")
    print(f"[KG] Sub-goal node probs: { {k: f'{v:.4f}' for k, v in node_probs.items()} }")
    return state, initial_bias


if __name__ == '__main__':
    graph = load_graph()

    class FakeOpt:
        def __init__(self, name): self.name = name

    options = [FakeOpt('Explore'), FakeOpt('GoToStairs'), FakeOpt('PickupItem'),
               FakeOpt('FindKey'), FakeOpt('OpenDoor')]

    state, init_bias = make_kg_path_state(graph, options, 'MiniHack-KeyRoom-S5-v0')

    print(f"\n[KG] Initial bias (step 0 — no sub-goals completed):")
    for opt, w in zip(options, init_bias):
        print(f"  {opt.name:20s}: {w:+.4f}")

    if state:
        print(f"\n[KG] Bias after key obtained (step 1):")
        state.current_step = 1
        b1 = state.get_bias()
        for opt, w in zip(options, b1):
            print(f"  {opt.name:20s}: {w:+.4f}")

        print(f"\n[KG] Bias after door opened (step 2):")
        state.current_step = 2
        b2 = state.get_bias()
        for opt, w in zip(options, b2):
            print(f"  {opt.name:20s}: {w:+.4f}")
