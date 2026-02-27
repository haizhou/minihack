"""
plot_results.py — 用已有的训练数据画学习曲线对比图
直接运行: python plot_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 从终端输出整理的数据 ──────────────────────────────────────────────────────

# PPO Baseline (MiniHack-Room-5x5-v0, 5000 steps)
ppo_episodes  = [10,20,30,40,50,60,70,80,90,100,
                 110,120,130,140,150,160,170,180,190,200,
                 210,220,230,240,250,260,270,280,290,300,
                 310,320,330]
ppo_returns   = [-0.229,0.784,0.992,0.993,0.997,0.997,0.996,0.996,0.996,0.997,
                  0.997,0.997,0.996,0.996,0.998,0.998,0.996,0.998,0.995,0.997,
                  0.996,0.998,0.997,0.997,0.998,0.998,0.997,0.998,0.996,0.996,
                  0.996,0.996,0.998]
ppo_steps     = [822,1278,1546,1808,1942,2073,2212,2360,2480,2602,
                 2714,2822,2936,3065,3172,3276,3395,3490,3622,3730,
                 3856,3953,4057,4168,4256,4359,4467,4570,4664,4783,
                 4896,5015,5116]

# Option-PPO (MiniHack-Room-5x5-v0, 5000 primitive steps)
opt_episodes  = [10,20,30,40,50,60,70]
opt_returns   = [0.376,0.383,-0.223,-0.020,0.381,-0.224,0.385]
opt_steps     = [827,1450,2260,2929,3538,4379,4937]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PPO Baseline vs Option-PPO on MiniHack-Room-5x5-v0",
             fontsize=13, fontweight='bold')

# ── Left: Return vs Episodes ──────────────────────────────────────────────────
ax1.plot(ppo_episodes, ppo_returns,  color='steelblue', lw=2, marker='o', ms=3,
         label='PPO Baseline')
ax1.plot(opt_episodes, opt_returns,  color='tomato',    lw=2, marker='s', ms=4,
         label='Option-PPO')
ax1.axhline(0, color='gray', lw=0.8, ls='--')
ax1.set_xlabel("Episode", fontsize=11)
ax1.set_ylabel("Mean Return (per 10 eps)", fontsize=11)
ax1.set_title("Learning Curve (by Episode)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, 1.2)

# Annotate convergence
ax1.annotate("Converges ~ep 30", xy=(30, 0.992), xytext=(60, 0.85),
             arrowprops=dict(arrowstyle='->', color='steelblue'),
             color='steelblue', fontsize=9)

# ── Right: Return vs Primitive Steps ─────────────────────────────────────────
ax2.plot(ppo_steps, ppo_returns, color='steelblue', lw=2, marker='o', ms=3,
         label='PPO Baseline')
ax2.plot(opt_steps, opt_returns, color='tomato',    lw=2, marker='s', ms=4,
         label='Option-PPO')
ax2.axhline(0, color='gray', lw=0.8, ls='--')
ax2.set_xlabel("Primitive Steps", fontsize=11)
ax2.set_ylabel("Mean Return (per 10 eps)", fontsize=11)
ax2.set_title("Sample Efficiency (by Primitive Steps)", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.5, 1.2)

# Note box
note = ("Note: Option-PPO makes fewer decisions\n"
        "(~70 option-choices vs ~330 action-choices)\n"
        "— shorter credit assignment horizon")
ax2.text(0.02, 0.05, note, transform=ax2.transAxes,
         fontsize=8.5, color='dimgray',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
print("Saved to comparison.png")
plt.show()
