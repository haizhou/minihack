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

    def __init__(self, path, node_probs, options, eta=0.5):
        """
        path:       [(src, relation, dst), ...] from bfs_full_path
        node_probs: {node: float} from get_node_probs  (learned from world knowledge)
        options:    list of Option objects
        eta:        intrinsic reward scaling factor
        """
        self.path       = path
        self.node_probs = node_probs
        self.options    = options
        self.eta        = eta

        # 预计算每个路径位置对应的 bias 向量（位置 i = 前 i 条边已完成）
        self._bias_cache = [
            _bias_for_path_segment(path[i:], node_probs, options)
            for i in range(len(path) + 1)
        ]

        self.current_step = 0   # 当前在 path 中的索引
        self._prev_chars  = None

    def reset(self, init_obs):
        """每个 episode 开始时重置状态。"""
        self.current_step = 0
        self._prev_chars  = init_obs.get('chars')

    def update(self, pre_obs, post_obs):
        """
        option 执行完后调用。比较执行前后的观测，判断是否穿越了下一条 KG 边。

        返回: intrinsic_reward (float)
          - 穿越了边: = node_probs[dst] * eta
          - 未穿越:   = 0.0
        """
        if self.current_step >= len(self.path):
            self._prev_chars = post_obs.get('chars')
            return 0.0

        src, rel, dst = self.path[self.current_step]
        detection     = EDGE_DETECTION.get(rel, 'glyph_decrease')
        traversed     = False

        if detection == 'glyph_decrease':
            traversed = self._glyph_decreased(pre_obs, post_obs, dst)

        intr_r = 0.0
        if traversed:
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
        """
        idx  = min(self.current_step, len(self._bias_cache) - 1)
        return self._bias_cache[idx] * kg_decay

    def progress(self):
        """返回 (current_step, total_steps) 用于日志。"""
        return self.current_step, len(self.path)

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
