
import re, json, numpy as np
from collections import defaultdict

def extract_entity(msg):
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
        if re.match(r'^[a-z] - ', msg):
            entity = extract_entity(msg)
            if entity:
                entity_count[entity] += 1
                graph['agent'].append({'relation': 'can_pickup', 'target': entity})
        elif 'tower of flame' in msg:
            graph['fire_trap'].append({'relation': 'causes', 'target': 'damage/burn'})
            if 'leather armor' in msg:
                graph['fire_trap'].append({'relation': 'damages', 'target': 'leather_armor'})
        elif 'the door opens' in msg:
            graph['door'].append({'relation': 'state_change', 'target': 'open'})
        elif 'the door resists' in msg:
            graph['key'].append({'relation': 'enables', 'target': 'door'})
        # 'the door closes' is intentionally omitted: door→closed is a dead-end
        # branch that misleads path-to-goal queries in kg_planner.
        elif 'fountain dries up' in msg:
            graph['fountain'].append({'relation': 'state_change', 'target': 'dried'})
        elif 'ishtar is displeased' in msg:
            graph['prayer'].append({'relation': 'causes', 'target': 'divine_displeasure'})

    clean_graph = {}
    for node, edges in graph.items():
        seen = set()
        clean_edges = []
        for e in edges:
            k = (e['relation'], e['target'])
            if k not in seen:
                seen.add(k)
                clean_edges.append(e)
        clean_graph[node] = clean_edges

    return clean_graph, entity_count

data = np.load('data/world_knowledge.npz', allow_pickle=True)
messages = data['messages'].tolist()
graph, counts = build_graph(messages)

print("=" * 50)
print("KNOWLEDGE GRAPH")
print("=" * 50)
for node, edges in graph.items():
    for e in edges:
        print(f"  {node:20s} --[{e['relation']}]--> {e['target']}")

with open('data/knowledge_graph.json', 'w') as f:
    json.dump({"graph": graph, 'entity_counts': dict(counts)}, f, indent=2)
print("Saved to data/knowledge_graph.json")
