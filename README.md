# MiniHack RL Demo — Options-PPO with KG-Guided Intrinsic Rewards

**FYP Project 72** | Supervisor: Eduardo Pignatelli

Hierarchical reinforcement learning on [MiniHack](https://github.com/facebookresearch/minihack) using the **Options framework** (semi-MDPs) combined with a self-updating **Knowledge Graph (KG)** that provides shaped intrinsic rewards. Trained against `MiniHack-KeyRoom-S5-v0`, a procedurally generated puzzle requiring the agent to find a key, unlock a door, and reach the staircase.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Meta-Policy (PPO)                        │
│         selects one of 5 macro-options each step            │
└────────────────────┬────────────────────────────────────────┘
                     │ option index
        ┌────────────▼──────────────────────────┐
        │           Options Layer                │
        │  0: ExploreOption       (random walk)  │
        │  1: GoToStairsOption    (BFS → '>')    │
        │  2: PickupItemOption    (BFS → '(')    │
        │  3: FindKeyOption       (BFS → key)    │
        │  4: OpenDoorOption      (BFS → '+')    │
        └────────────┬──────────────────────────┘
                     │ primitive actions (× up to 30 steps)
        ┌────────────▼──────────────────────────┐
        │       MiniHack-KeyRoom-S5-v0           │
        │   Action space: 10 (8 dirs + PICKUP    │
        │                     + APPLY)           │
        └───────────────────────────────────────┘
                     │ observations
        ┌────────────▼──────────────────────────┐
        │      KGPathState (kg_planner.py)       │
        │  Tracks agent progress along path:     │
        │  agent → key → door → open → stairs   │
        │  Issues intrinsic reward on traversal  │
        │  Self-updates node probs from wins     │
        └───────────────────────────────────────┘
```

---

## Project Structure

```
minihack-demo/
├── options_ppo.py          # Main training script: Options-PPO + KG intrinsic rewards
├── kg_planner.py           # KGPathState: dynamic KG tracker + self-updating priors
├── ppo_baseline.py         # Flat PPO on primitive actions (comparison baseline)
├── build_knowledge_graph.py# Builds KG from collected world-knowledge transitions
├── collect_transitions.py  # Auto-explorer to collect NetHack messages → world_knowledge.npz
│
├── data/
│   ├── knowledge_graph.json     # KG definition (nodes, edges, entity counts)
│   ├── knowledge_graph.png      # Visualized directed graph
│   ├── kg_learned_probs.json    # Learned node probabilities (persisted across runs)
│   └── world_knowledge.npz      # Transition dataset collected from DenseItemRoom env
│
├── models/                      # Saved model checkpoints (.pt)
│   ├── ppo_baseline.pt
│   ├── s15.pt / s15_fresh.pt
│   └── s17_fresh_run.pt         # Latest trained Options-PPO model
│
├── analysis/
│   ├── plot_s17.py              # S17 run visualization script
│   └── s17_training_analysis.png# 4-panel training analysis figure
│
├── figures/
│   ├── comparison.png           # Baseline vs Options return comparison
│   ├── comparison_full.png      # Extended comparison with SPS
│   └── kg_options_learning_curve.png
│
├── logs/                        # Raw training logs
└── requirements.txt
```

---

## Options

| # | Name | Behaviour | Terminates when |
|---|------|-----------|-----------------|
| 0 | ExploreOption | Random walk with visited-cell tracking | 30 primitive steps |
| 1 | GoToStairsOption | BFS toward `>` (staircase glyph) | Reached staircase or 30 steps |
| 2 | PickupItemOption | BFS toward nearest `(` item, issue PICKUP | Item picked or 30 steps |
| 3 | FindKeyOption | BFS toward "Master Key of Thievery" `(` | Key picked up or 30 steps |
| 4 | OpenDoorOption | BFS toward `+` door, issue APPLY to unlock | Door opens (no `+` visible) or 30 steps |

The meta-policy (PPO) selects options; each option executes up to `MAX_OPTION_STEPS=30` primitive actions. Returns are computed at the option level (SMDP discounting).

---

## Knowledge Graph

The KG encodes prior world knowledge as a directed graph with probabilistic nodes:

```
agent --[can_pickup]--> key  (prob ≈ 0.57)
key   --[enables]---> door   (prob ≈ 0.57)
door  --[state_change]--> open   (auto-fires)
open  --[discovered]--> stairs  (prob ≈ 0.57, discovered after first win)
```

**KGPathState** (`kg_planner.py`) maintains per-episode state:
- Tracks which edge was last traversed
- Detects edge traversal by monitoring glyph counts (pickup/open = glyph disappears)
- Issues intrinsic reward `= node_prob × η × kg_decay` on each edge traversal
- Self-updates node probabilities via EMA (lr=0.01) from observed successes
- Learns ordering constraints (e.g., door-before-key is penalised after repeated failures)

**KG intrinsic reward decay:** `kg_decay` starts at 1.0, decays to a floor of 0.15 over 70% of training, so the agent transitions from KG-guided exploration to extrinsic-reward-driven exploitation.

---

## Setup

**Requirements:** Linux or WSL2, Python 3.9, Conda

```bash
# 1. Create environment
conda create -n minihack python=3.9 -y
conda activate minihack

# 2. System dependencies (NLE build)
sudo apt-get install -y build-essential cmake ninja-build \
    libboost-all-dev libbz2-dev zlib1g-dev flex bison

# 3. Python packages
pip install nle minihack torch gymnasium numpy matplotlib
# or:
pip install -r requirements.txt
```

---

## Usage

### Train Options-PPO (recommended)
```bash
python options_ppo.py \
    --env MiniHack-KeyRoom-S5-v0 \
    --steps 200000 \
    --save_path models/my_run.pt
```

### Train PPO Baseline (primitive actions)
```bash
python ppo_baseline.py \
    --env MiniHack-KeyRoom-S5-v0 \
    --steps 200000 \
    --save_path models/baseline.pt
```

### Rebuild Knowledge Graph
```bash
# Step 1: collect world-knowledge transitions
python collect_transitions.py

# Step 2: build KG from collected messages
python build_knowledge_graph.py
```

### Visualize Training Results
```bash
python analysis/plot_s17.py
# outputs: analysis/s17_training_analysis.png
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.99 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE lambda |
| `CLIP_EPS` | 0.2 | PPO clip epsilon |
| `ROLLOUT_LEN` | 128 | Steps per rollout collection |
| `N_EPOCHS` | 6 | PPO update epochs per rollout |
| `ENTROPY_COEF` | 0.05→0.005 | Entropy bonus (decays over training) |
| `ETA_REWARD` | 0.1 | Intrinsic reward scale |
| `kg_decay` | 1.0→0.15 | KG reward decay (floor at 0.15 after 70% of steps) |
| `MAX_OPTION_STEPS` | 30 | Max primitive steps per option |
| `reward_win` | 5.0 | Extrinsic reward on reaching stairs |

---

## S17 Training Results (200K steps)

| Metric | Value |
|--------|-------|
| Total primitive steps | ~199,343 |
| Total episodes | ~1,404 |
| Final mean return (20-ep window) | **2.555** |
| Peak SPS | ~1,175 |
| Dominant option | OpenDoor (11,399 uses, ~55%) |
| KG node probs (converged) | key/door/stairs ≈ 0.57 |

Training phases:
1. **Early (ep 1–40):** KG dependency learning, SPS ramp-up from 191 → 423
2. **Mid (ep 40–260):** Volatile returns (−1 to +4), SPS rising to ~680
3. **Late (ep 260–800):** Stabilising return ~1.5–2.5, KG probs converging, SPS → 1050
4. **Final (ep 800–1404):** Consistent returns 2–2.6, SPS plateau ~1,175

---

## Implementation Notes

**NLE shared C buffer bug (critical):**
NLE observation arrays are views into a shared C buffer. Storing `pre_obs = obs` copies the
reference, not the data — after `execute_option()` runs multiple `env.step()` calls the buffer
is overwritten. Fix (`options_ppo.py` line ~534):
```python
pre_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
```

**WALKABLE set:** `set(map(ord, ' .@<>_#('))` — includes items `(` but excludes `+` (door) and
wall characters. Options navigate through walkable cells; `OpenDoorOption` moves adjacent to the
door then issues APPLY.

**Floor key detection:** The starting inventory always has items in slots a–f (sword, daggers,
armour, etc.). The floor key "Master Key of Thievery" lands in slot `g` (index 6). The
`key_in_inventory()` helper searches by substring `"key"` in item descriptions.

---

## References

- [MiniHack: A Sandbox for Open-Ended Reinforcement Learning Research](https://arxiv.org/abs/2109.13523)
- [NLE: The NetHack Learning Environment](https://arxiv.org/abs/2006.13760)
- [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL (Sutton et al., 1999)](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
- [PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
