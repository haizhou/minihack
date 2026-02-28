"""Quick sanity-check for kg_planner.py"""
import numpy as np
import torch
from kg_planner import load_graph, get_option_weights, dynamic_option_bias, NODE_TO_CHAR

# ── Fake options matching the names used in options_ppo.py ────────────────────
class FakeOpt:
    def __init__(self, name): self.name = name

OPTIONS = [
    FakeOpt("Explore"),
    FakeOpt("GoToStairs"),
    FakeOpt("PickupItem"),
    FakeOpt("FindKey"),
    FakeOpt("OpenDoor"),
]

ENV = "MiniHack-KeyRoom-S5-v0"

graph = load_graph()

# ── Test 1: static option weights ─────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — Static option weights (get_option_weights)")
print("=" * 60)
weights = get_option_weights(graph, OPTIONS, ENV)
for opt, w in zip(OPTIONS, weights):
    bar = "+" * int(abs(w) * 5) if w > 0 else "-" * int(abs(w) * 5)
    print(f"  {opt.name:15s}: {w:+.4f}  {bar}")

print()
assert weights[3] > 0, "FindKey should be boosted"
assert weights[4] > 0, "OpenDoor should be boosted"
assert weights[0] < 0, "Explore should be penalised"
print("  [PASS] FindKey boosted, OpenDoor boosted, Explore penalised\n")

# ── Test 2: dynamic bias — key NOT visible ────────────────────────────────────
print("=" * 60)
print("TEST 2 — dynamic_option_bias, key NOT visible")
print("=" * 60)
ROWS, COLS = 21, 79
chars_no_key = np.full((ROWS, COLS), ord('.'), dtype=np.uint8)
obs_no_key = {"glyphs": np.zeros((ROWS, COLS), dtype=np.int32), "chars": chars_no_key}
bias_no_key = dynamic_option_bias(graph, obs_no_key, OPTIONS, ENV, weights.clone())
for opt, w in zip(OPTIONS, bias_no_key):
    print(f"  {opt.name:15s}: {w:+.4f}")
findkey_no_key = bias_no_key[3].item()
print()

# ── Test 3: dynamic bias — key IS visible ─────────────────────────────────────
print("=" * 60)
print("TEST 3 — dynamic_option_bias, key VISIBLE (char '(' planted at [10,10])")
print("=" * 60)
chars_key = chars_no_key.copy()
chars_key[10, 10] = ord('(')          # plant a key glyph
obs_key = {"glyphs": np.zeros((ROWS, COLS), dtype=np.int32), "chars": chars_key}
bias_key = dynamic_option_bias(graph, obs_key, OPTIONS, ENV, weights.clone())
for opt, w in zip(OPTIONS, bias_key):
    print(f"  {opt.name:15s}: {w:+.4f}")
findkey_visible = bias_key[3].item()
print()

# ── Verify the 5x boost ───────────────────────────────────────────────────────
print("=" * 60)
print("BOOST COMPARISON")
print("=" * 60)
delta = findkey_visible - findkey_no_key
print(f"  FindKey (key hidden) : {findkey_no_key:+.4f}")
print(f"  FindKey (key visible): {findkey_visible:+.4f}")
print(f"  Delta                : {delta:+.4f}")

from kg_planner import get_node_probs, OPTION_TO_NODE
probs, _ = get_node_probs(graph, 'agent', 'open')
key_prob = probs.get('key', 0.0)

# After the deduplication fix, each node is boosted at most once per option.
# Delta should be exactly p_key * (5.0 - 1.0) = p_key * 4.0 regardless of
# how many keywords happen to substring-match the option name.
expected_extra = key_prob * 4.0
print(f"\n  Expected delta (p_key * 4.0 = {key_prob:.4f} * 4): {expected_extra:+.4f}")
assert abs(delta - expected_extra) < 1e-5, f"Expected delta {expected_extra:.4f}, got {delta:.4f}"
print("\n  [PASS] Visibility boost is exactly 5x (node-level deduplication working correctly)")
