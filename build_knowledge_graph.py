
import re, json, numpy as np
from collections import defaultdict

def extract_entity(msg):
    # 从背包消息提取物品名
    m = re.match(r'^[a-z] - (?:an? )?(?:uncursed |cursed |blessed )?(?:burnt |very burnt |thoroughly burnt )?(?:<bonus> )?(.*?)(?:\..*)?$', msg)
    if m:
        item = m.group(1).strip()
        item = re.sub(r'\(.*\)', '', item).strip()
        return item
    return None

def build_graph(messages):
    graph = defaultdict(list)
    entity_count = defaultdict(int)

    for msg in messages:
        msg = msg.strip()

        # 捡到物品
        if re.match(r'^[a-z] - ', msg):
            entity = extract_entity(msg)
            if entity:
                entity_count[entity] += 1
                graph['agent'].append({'relation': 'can_pickup', 'target': entity})

        # 火焰陷阱
        elif 'tower of flame' in msg:
            graph['fire_trap'].append({'relation': 'causes', 'target': 'damage/burn'})
            if 'leather armor' in msg:
                graph['fire_trap'].append({'relation': 'damages', 'target': 'leather_armor'})

        # 门
        elif 'the door opens' in msg:
            graph['door'].append({'relation': 'state_change', 'target': 'open'})
        elif 'the door resists' in msg:
            graph['door'].append({'relation': 'requires', 'target': 'key'})
        elif 'the door closes' in msg:
            graph['door'].append({'relation': 'state_change', 'target': 'closed'})

        # 喷泉
        elif 'fountain dries up' in msg:
            graph['fountain'].append({'relation': 'state_change', 'target': 'dried'})

        # 神明不满
        elif 'ishtar is displeased' in msg:
            graph['prayer'].append({'relation': 'causes', 'target': 'divine_displeasure'})

    # 去重
    clean_graph = {}
    for node, edges in graph.items():
        seen = set()
        clean_edges = []
        for e in edges:
            key = (e['relation'], e['target'])
            if key not in seen:
                seen.add(key)
                clean_edges.append(e)
        clean_graph[node] = clean_edges

    return clean_graph, entity_count

# 加载数据
data = np.load('data/world_knowledge.npz', allow_pickle=True)
messages = data['messages'].tolist()

graph, counts = build_graph(messages)

print("=" * 50)
print("KNOWLEDGE GRAPH")
print("=" * 50)
for node, edges in graph.items():
    for e in edges:
        print(f"  {node:20s} --[{e['relation']}]--> {e['target']}")

print()
print("=" * 50)
print("ENTITY FREQUENCY (what agent picked up most)")
print("=" * 50)
for entity, count in sorted(counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {entity:30s}: {count}")

# 保存
with open('data/knowledge_graph.json', 'w') as f:
    json.dump({"graph": graph, 'entity_counts': dict(counts)}, f, indent=2)
print()
print("Saved to data/knowledge_graph.json")
