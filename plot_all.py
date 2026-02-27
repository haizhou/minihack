"""
plot_all.py — 综合对比图：5x5 简单环境 + KeyRoom 困难环境
运行: python plot_all.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ── 数据 ──────────────────────────────────────────────────────────────────────

# 5x5 PPO Baseline
ppo_5x5_eps  = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,
                160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330]
ppo_5x5_ret  = [-0.229,0.784,0.992,0.993,0.997,0.997,0.996,0.996,0.996,0.997,
                 0.997,0.997,0.996,0.996,0.998,0.998,0.996,0.998,0.995,0.997,
                 0.996,0.998,0.997,0.997,0.998,0.998,0.997,0.998,0.996,0.996,
                 0.996,0.996,0.998]
ppo_5x5_steps= [822,1278,1546,1808,1942,2073,2212,2360,2480,2602,
                 2714,2822,2936,3065,3172,3276,3395,3490,3622,3730,
                 3856,3953,4057,4168,4256,4359,4467,4570,4664,4783,
                 4896,5015,5116]

# 5x5 Option-PPO
opt_5x5_eps  = [10,20,30,40,50,60,70]
opt_5x5_ret  = [0.376,0.383,-0.223,-0.020,0.381,-0.224,0.385]
opt_5x5_steps= [827,1450,2260,2929,3538,4379,4937]

# KeyRoom PPO Baseline
ppo_kr_eps   = [10,20,30,40,50,60,70,80,90,100]
ppo_kr_ret   = [-0.919,-1.124,-0.906,-1.104,-1.098,-1.126,-1.107,-1.098,-1.110,-0.894]
ppo_kr_steps = [1942,3942,5845,7845,9845,11845,13845,15845,17845,19773]

# KeyRoom Option-PPO
opt_kr_eps   = [10,20,30,40,50,60,70,80,90,100]
opt_kr_ret   = [-1.086,-1.088,-1.087,-1.084,-1.086,-1.086,-1.088,-1.081,-1.082,-1.088]
opt_kr_steps = [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000]

# Option usage in KeyRoom
option_names  = ['Explore', 'NavigateToStaircase', 'PickupItem']
option_counts = [195, 233, 272]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Options for Temporal Credit Assignment — MiniHack Demo\n"
             "FYP Project 72 | Supervisor: Eduardo Pignatelli",
             fontsize=13, fontweight='bold', y=0.98)

# Layout: 2x2 + 1 bar chart
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 4)
ax4 = fig.add_subplot(2, 3, 5)
ax5 = fig.add_subplot(2, 3, (3, 6))

C_PPO = 'steelblue'
C_OPT = 'tomato'

# ── Top left: 5x5 by episode ──────────────────────────────────────────────────
ax1.plot(ppo_5x5_eps, ppo_5x5_ret, color=C_PPO, lw=2, marker='o', ms=2, label='PPO Baseline')
ax1.plot(opt_5x5_eps, opt_5x5_ret, color=C_OPT, lw=2, marker='s', ms=4, label='Option-PPO')
ax1.axhline(0, color='gray', lw=0.7, ls='--')
ax1.set_title("Simple Env (5×5) — by Episode", fontsize=10)
ax1.set_xlabel("Episode"); ax1.set_ylabel("Mean Return")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_ylim(-0.6, 1.3)
ax1.annotate("Converges ep~30", xy=(30,0.992), xytext=(80,0.7),
             arrowprops=dict(arrowstyle='->', color=C_PPO), color=C_PPO, fontsize=8)

# ── Top right: 5x5 by steps ───────────────────────────────────────────────────
ax2.plot(ppo_5x5_steps, ppo_5x5_ret, color=C_PPO, lw=2, marker='o', ms=2, label='PPO Baseline')
ax2.plot(opt_5x5_steps, opt_5x5_ret, color=C_OPT, lw=2, marker='s', ms=4, label='Option-PPO')
ax2.axhline(0, color='gray', lw=0.7, ls='--')
ax2.set_title("Simple Env (5×5) — by Primitive Steps", fontsize=10)
ax2.set_xlabel("Primitive Steps"); ax2.set_ylabel("Mean Return")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.6, 1.3)

# ── Bottom left: KeyRoom by episode ───────────────────────────────────────────
ax3.plot(ppo_kr_eps, ppo_kr_ret, color=C_PPO, lw=2, marker='o', ms=4, label='PPO Baseline')
ax3.plot(opt_kr_eps, opt_kr_ret, color=C_OPT, lw=2, marker='s', ms=4, label='Option-PPO')
ax3.axhline(0, color='gray', lw=0.7, ls='--')
ax3.set_title("Hard Env (KeyRoom-S5) — by Episode", fontsize=10)
ax3.set_xlabel("Episode"); ax3.set_ylabel("Mean Return")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_ylim(-1.5, 0.3)
ax3.text(0.05, 0.08, "Both struggle — more steps needed\nto demonstrate option advantage",
         transform=ax3.transAxes, fontsize=8, color='dimgray',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ── Bottom middle: KeyRoom by steps ───────────────────────────────────────────
ax4.plot(ppo_kr_steps, ppo_kr_ret, color=C_PPO, lw=2, marker='o', ms=4, label='PPO Baseline')
ax4.plot(opt_kr_steps, opt_kr_ret, color=C_OPT, lw=2, marker='s', ms=4, label='Option-PPO')
ax4.axhline(0, color='gray', lw=0.7, ls='--')
ax4.set_title("Hard Env (KeyRoom-S5) — by Primitive Steps", fontsize=10)
ax4.set_xlabel("Primitive Steps"); ax4.set_ylabel("Mean Return")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3); ax4.set_ylim(-1.5, 0.3)

# SPS annotation
ax4.text(0.05, 0.85, "Option-PPO SPS: ~700\nBaseline SPS: ~214",
         transform=ax4.transAxes, fontsize=8, color='dimgray',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# ── Right: Option usage bar chart ─────────────────────────────────────────────
colors = ['#66b3ff', '#ff9999', '#99ff99']
bars = ax5.bar(option_names, option_counts, color=colors, edgecolor='gray', linewidth=0.8)
ax5.set_title("Option Usage — KeyRoom-S5\n(Meta-policy learned preference)", fontsize=10)
ax5.set_ylabel("Times Selected")
ax5.grid(True, axis='y', alpha=0.3)

for bar, cnt in zip(bars, option_counts):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(cnt), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax5.text(0.5, 0.15,
         "PickupItem selected most (272×)\n→ Meta-policy learns item collection\n   is more rewarding than random walk",
         transform=ax5.transAxes, ha='center', fontsize=9, color='dimgray',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("comparison_full.png", dpi=150, bbox_inches='tight')
print("Saved to comparison_full.png")
plt.show()
