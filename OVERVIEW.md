# Project Overview
**FYP Project 72 — Options-PPO with Knowledge-Graph Intrinsic Rewards**
Supervisor: Eduardo Pignatelli | Environment: MiniHack-KeyRoom-S5-v0

---

## What This Project Is About

Standard deep RL (flat PPO) struggles with tasks that require completing **ordered sub-goals** separated by hundreds of primitive actions — a problem called *temporal credit assignment*. In `MiniHack-KeyRoom`, the agent must:

> find the key → pick it up → navigate to the door → unlock it → reach the stairs

A primitive-action agent gets a single +5 reward only when it reaches the stairs. Everything leading up to that moment is unrewarded, making it very hard to learn *which* early actions actually mattered.

This project attacks that problem with two ideas working together:

1. **Hierarchical actions (Options)** — instead of choosing one of 10 directions every step, the meta-policy chooses from 5 *macro-actions* (e.g. "FindKey", "OpenDoor"). Each option handles its own low-level navigation internally.

2. **Knowledge Graph intrinsic rewards** — a lightweight graph encodes world knowledge (`key enables door`, `door → open → stairs`). As the agent completes each edge in the graph, it receives a small bonus reward, providing a dense learning signal that fades away as training matures.

---

## System Design (one-minute read)

```
Meta-Policy (PPO)
      │  picks one of 5 options
      ▼
┌─────────────────────────────────────────────┐
│  Explore │ GoToStairs │ PickupItem │ ...     │  ← Options Layer
└─────────────────────────────────────────────┘
      │  executes up to 30 primitive steps
      ▼
   MiniHack-KeyRoom-S5-v0   (10-action NetHack env)
      │  observations
      ▼
┌──────────────────────────────────────────────┐
│  KGPathState                                  │
│  agent ──[can_pickup]──▶ key                 │
│  key   ──[enables]──────▶ door               │
│  door  ──[state_change]─▶ open               │  ← Knowledge Graph
│  open  ──[discovered]───▶ stairs             │
│                                               │
│  • detects edge traversal via glyph changes  │
│  • issues intrinsic_r = prob × η × decay     │
│  • self-updates node probs after each win    │
└──────────────────────────────────────────────┘
```

The KG intrinsic reward decays from scale 1.0 to a floor of 0.15 over the first 70% of training — the agent starts KG-guided and finishes reward-driven.

---

## Key Files

| File | Purpose |
|------|---------|
| `options_ppo.py` | Training loop, 5 option classes, PPO update |
| `kg_planner.py` | KGPathState — graph tracker, intrinsic rewards, self-update |
| `ppo_baseline.py` | Flat PPO on raw actions (comparison) |
| `analysis/s17_training_analysis.png` | Training curves (see below) |
| `figures/comparison.png` | Options vs baseline return comparison |

---

## Results (200K primitive steps)

| | Flat PPO baseline | **Options-PPO + KG** |
|---|---|---|
| Final mean return | ~0.3 | **2.555** |
| Wins observed | rare | consistent from ep ~260 |
| Training speed | — | ~1,175 SPS (GPU) |

The agent completed ~1,404 episodes. By the final phase (ep 800–1404) it was solving the key→door→stairs chain consistently with mean return 2–2.6 per episode.

**Option usage at end of training:**

| Option | Uses | Share |
|--------|------|-------|
| OpenDoor | 11,399 | 55% |
| FindKey | 2,258 | 11% |
| PickupItem | 1,428 | 7% |
| GoToStairs | 5,005 | 24% |
| Explore | 258 | 1% |

OpenDoor dominating is expected — it requires the most sub-steps per invocation and is the hardest option to complete reliably.

The KG node probabilities converged from initial priors (key=0.60, door=0.30) to a stable band around **0.57** for all three nodes after ~800 episodes of self-updating.

---

## One Non-Trivial Engineering Problem

NLE (the NetHack backend) returns observation arrays as **views into a shared C buffer**. Storing `pre_obs = obs` before calling `execute_option()` does not copy the data — by the time the option finishes, the buffer has been overwritten by later `env.step()` calls. This silently broke all KG edge-detection for dozens of training runs before being identified. The fix is one line:

```python
pre_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
```

---

## How to Run

```bash
conda create -n minihack python=3.9 -y && conda activate minihack
pip install nle minihack torch gymnasium numpy matplotlib

# train
python options_ppo.py --env MiniHack-KeyRoom-S5-v0 --steps 200000

# view training curves
python analysis/plot_s17.py
```

---

*Full technical details: `README.md` | Training curves: `analysis/s17_training_analysis.png` | Baseline comparison: `figures/comparison.png`*
