import json, torch
from collections import deque

def load_graph(path='data/knowledge_graph.json'):
    with open(path) as f:
        return json.load(f)['graph']

def bfs_full_path(graph, start, goal):
    queue = deque([(start, [])])
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

    path_nodes = [t for s, r, t in path if t != goal]
    n_path = len(path_nodes)

    path_scores = torch.tensor([1.0 / (i + 1) for i in range(n_path)])
    path_probs  = path_scores / path_scores.sum() * 0.9
    other_prob  = 0.1 / n_other_estimate

    probs = {node: p for node, p in zip(path_nodes, path_probs.tolist())}
    return probs, path

def get_option_weights(graph, options, env_name, start='agent', goal='open'):
    if 'KeyRoom' not in env_name:
        return torch.zeros(len(options))

    probs, path = get_node_probs(graph, start, goal)
    weights = torch.zeros(len(options))

    print(f"[KG] Optimal path:")
    for s, r, t in path:
        print(f"[KG]   {s:20s} -[{r}]-> {t}")

    print(f"[KG] Node probabilities:")
    for node, p in probs.items():
        print(f"[KG]   {node:20s}: p={p:.4f}")

    for i, opt in enumerate(options):
        name = opt.name.lower()
        for node, p in probs.items():
            if node == 'key':
                if 'find' in name or 'key' in name or 'pickup' in name:
                    weights[i] += p * 3.0
            if node == 'door':
                if 'door' in name or 'open' in name:
                    weights[i] += p * 3.0
        if 'explore' in name:
            weights[i] -= 0.5

    return weights

if __name__ == '__main__':
    graph = load_graph()

    class FakeOpt:
        def __init__(self, name): self.name = name

    options = [FakeOpt('Explore'), FakeOpt('GoToStairs'), FakeOpt('PickupItem'),
               FakeOpt('FindKey'), FakeOpt('OpenDoor')]

    weights = get_option_weights(graph, options, 'MiniHack-KeyRoom-S5-v0')
    print(f"\n[KG] Option weights:")
    for opt, w in zip(options, weights):
        print(f"  {opt.name:20s}: {w:+.4f}")

# 节点对应的 glyph 字符
NODE_TO_CHAR = {
    'key':    {ord('(')},
    'door':   {ord('+'), ord('|'), ord('-')},
    'open':   {ord('.')},
    'stairs': {ord('>')},
}

# option 名称对应负责到达哪个节点
OPTION_TO_NODE = {
    'findkey':   'key',
    'find':      'key',
    'opendoor':  'door',
    'open':      'door',
    'gotostairs':'stairs',
    'stair':     'stairs',
    'pickup':    'key',
}

def dynamic_option_bias(graph, obs, options, env_name, base_weights, kg_decay=1.0):
    from nle import nethack
    bias = base_weights.clone()

    if 'KeyRoom' not in env_name:
        return bias

    glyphs = obs.get('glyphs')
    if glyphs is None:
        return bias

    # 扫描视野内可见的字符（直接用 chars 数组）
    chars = obs.get('chars')
    if chars is None:
        return bias
    visible_chars = set()
    for r in range(chars.shape[0]):
        for c in range(chars.shape[1]):
            visible_chars.add(int(chars[r, c]))

    # 从路径获取每步目标节点和概率
    probs, path = get_node_probs(graph, 'agent', 'open')
    # path = [(agent, can_pickup, key), (key, enables, door), (door, state_change, open)]

    for step_idx, (src, rel, tgt) in enumerate(path):
        if tgt == 'open':
            continue
        tgt_chars = NODE_TO_CHAR.get(tgt, set())
        tgt_visible = bool(tgt_chars & visible_chars)
        step_prob = probs.get(tgt, 0.0)

        for i, opt in enumerate(options):
            name = opt.name.lower()
            # Collect matched target nodes into a set to deduplicate — multiple
            # keywords (e.g. 'findkey' and 'find') can map to the same node, and
            # we want to apply the boost once per node, not once per keyword.
            matched_nodes = {node for kw, node in OPTION_TO_NODE.items()
                             if kw in name and node == tgt}
            if matched_nodes:
                if tgt_visible:
                    # 目标在视野内，强烈加强；随 kg_decay 衰减至零
                    bias[i] += step_prob * 5.0 * kg_decay
                else:
                    # 目标不在视野，按路径概率适当加强；随 kg_decay 衰减至零
                    bias[i] += step_prob * 1.0 * kg_decay

    return bias
