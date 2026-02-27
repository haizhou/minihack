
import json
import torch
import numpy as np

def load_graph(path='data/knowledge_graph.json'):
    with open(path) as f:
        return json.load(f)['graph']

def query_chain(graph, start, relation, then_target_relation, final):
    """
    查询两跳关系链：
    start -[relation]-> X -[then_target_relation]-> final
    返回中间节点 X 的列表
    """
    mid_nodes = []
    for e in graph.get(start, []):
        if e['relation'] == relation:
            mid = e['target']
            for e2 in graph.get(final, []):
                if e2['relation'] == then_target_relation and e2['target'] == mid:
                    mid_nodes.append(mid)
    return mid_nodes

def get_option_prior(graph, options, env_name):
    """
    根据知识图谱和环境，返回每个 option 的先验 logit 偏置
    正数 = 更倾向选择，负数 = 不倾向
    """
    n = len(options)
    bias = torch.zeros(n)

    if 'KeyRoom' in env_name:
        # 查询：agent can_pickup X, door requires X → X 是关键物品
        key_items = query_chain(graph, 'agent', 'can_pickup', 'requires', 'door')
        print(f"[KG] Door requires items that agent can pickup: {key_items}")

        for i, opt in enumerate(options):
            name = opt.name.lower()
            # FindKey option 优先
            if any(item in name for item in ['key', 'find']):
                bias[i] += 2.0
                print(f"[KG] Boosting option '{opt.name}' (finds key)")
            # OpenDoor option 次之
            elif 'door' in name or 'open' in name:
                bias[i] += 1.0
                print(f"[KG] Boosting option '{opt.name}' (opens door)")
            # GoToStairs 也有用
            elif 'stair' in name:
                bias[i] += 0.5
                print(f"[KG] Boosting option '{opt.name}' (goes to goal)")
            # Explore 降低优先级
            elif 'explore' in name:
                bias[i] -= 0.5
                print(f"[KG] Reducing option '{opt.name}' (pure explore)")

    return bias

if __name__ == '__main__':
    graph = load_graph()

    # 模拟 options
    class FakeOpt:
        def __init__(self, name): self.name = name
    options = [FakeOpt('Explore'), FakeOpt('GoToStairs'), FakeOpt('PickupItem'),
               FakeOpt('FindKey'), FakeOpt('OpenDoor')]

    bias = get_option_prior(graph, options, 'MiniHack-KeyRoom-S5-v0')
    print()
    print("Option priors:")
    for opt, b in zip(options, bias):
        print(f"  {opt.name:20s}: {b:+.1f}")
