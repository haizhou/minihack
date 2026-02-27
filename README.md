# MiniHack RL Demo — Options for Temporal Credit Assignment

FYP Project 72 | Supervisor: Eduardo Pignatelli

## Setup (Linux / WSL2)
```bash
conda create -n minihack python=3.9 -y
conda activate minihack
sudo apt-get install -y build-essential cmake ninja-build libboost-all-dev libbz2-dev zlib1g-dev flex bison
pip install nle minihack torch gymnasium numpy matplotlib
```

## Run
```bash
# PPO baseline (primitive actions)
python ppo_baseline.py --env MiniHack-Room-5x5-v0 --steps 100000

# Option-level PPO (macro actions)
python options_ppo.py --env MiniHack-Room-5x5-v0 --steps 100000
```

## Options

| # | Name | Behaviour | Terminates when |
|---|------|-----------|-----------------|
| 0 | ExploreOption | Random walk | 30 primitive steps |
| 1 | NavigateToStaircase | BFS toward `>` | Reached staircase or 30 steps |
| 2 | PickupItem | BFS toward nearest item, pick up | Item picked or 30 steps |
