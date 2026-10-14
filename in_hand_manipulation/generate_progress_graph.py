#!/usr/bin/env python3
"""
generate_progress_graph.py
Generates high-resolution graphs of success and drop rates since 08:30 today (Iter 20,500 - 30,300+).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = "/home/akshat/.gemini/antigravity/brain/8ec2314e-e375-4c1b-9cd6-607d3833d9c7"
OUTPUT_PNG = os.path.join(ARTIFACT_DIR, "run_since_830_progress.png")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. Parse Benchmark History (500-episode evals)
active_dir = os.path.join(PROJECT_ROOT, "fine_tune_training")
if not os.path.exists(os.path.join(active_dir, "progress.csv")):
    active_dir = os.path.join(PROJECT_ROOT, "stage2_unified")

eval_path = os.path.join(active_dir, "eval_benchmark_history.csv")
b_iters = []
b_succ = []
b_drop = []
b_ret = []
b_clos = []
b_score = []
b_times = []

if os.path.exists(eval_path):
    with open(eval_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    for l in lines[1:]:
        p = l.split(",")
        if len(p) >= 10:
            ts = p[9]
            b_iters.append(int(p[0]))
            b_succ.append(float(p[2]) * 100)
            b_clos.append(float(p[3]) * 100)
            b_drop.append(float(p[4]) * 100)
            b_ret.append(float(p[5]) * 100)
            b_score.append(float(p[7]))
            try:
                b_times.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass

# 2. Parse Continuous Progress CSV (Rollouts)
prog_path = os.path.join(active_dir, "progress.csv")
r_iters = []
r_succ = []
r_drop = []
r_ret = []
r_clos = []
r_emi = []

if os.path.exists(prog_path):
    with open(prog_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    for l in lines[1:]:
        p = l.split(",")
        it = int(p[0])
        if it >= 20500:
            r_iters.append(it)
            r_succ.append(float(p[4]) * 100)
            r_drop.append(float(p[5]) * 100)
            r_ret.append(float(p[6]) * 100)
            r_clos.append(float(p[7]) * 100)
            r_emi.append(float(p[8]))

# Apply smoothing for rollout curves (rolling window of 25 iters)
def smooth(y, box_pts=25):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

# 3. Create Multi-Panel Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=200)
plt.subplots_adjust(hspace=0.25)

# Styling
bg_color = "#FAFAFA"
grid_color = "#E0E0E0"

# --- Panel 1: Benchmark Evaluations (500 Test Episodes @ 10% Error) ---
ax1 = axes[0]
ax1.set_facecolor(bg_color)
ax1.grid(True, linestyle="--", alpha=0.6, color=grid_color)
ax1.set_title("1. Continuous Benchmark Evaluations (500 Test Episodes per Milestone Across All Objects)", fontsize=13, fontweight="bold", pad=10)

ax1.plot(b_iters, b_succ, color="#1B5E20", linewidth=2.4, marker="o", markersize=4, label="Benchmark Success Rate (% at 10% Tol)")
ax1.plot(b_iters, b_ret, color="#0D47A1", linewidth=2.0, marker="s", markersize=3, label="Pre-Goal Retention Rate (%)")
ax1.plot(b_iters, b_clos, color="#6A1B9A", linewidth=1.8, linestyle="-.", label="Rotational Closure Rate (%)")
ax1.plot(b_iters, b_drop, color="#C62828", linewidth=1.8, linestyle="--", label="Pre-Goal Drop Rate (%)")

# Highlight Launch of Option A & Peak
opt_a_iter = 20659
ax1.axvline(x=opt_a_iter, color="#E65100", linestyle=":", linewidth=2.0, label="Option A Launched (08:40)")
ax1.text(opt_a_iter + 80, 20, "Option A Launch\n(w_prog=12, w1=2.5,\n10% tol locked)", color="#E65100", fontsize=9, fontweight="bold", bbox=dict(facecolor="#FFF3E0", edgecolor="#E65100", boxstyle="round,pad=0.3"))

# Highlight Peak Model
if b_succ:
    max_succ_idx = np.argmax(b_succ)
    p_it = b_iters[max_succ_idx]
    p_sc = b_succ[max_succ_idx]
    p_cl = b_clos[max_succ_idx]
    p_rt = b_ret[max_succ_idx]
    ax1.plot(p_it, p_sc, marker="*", color="#D84315", markersize=14, zorder=5)
    ax1.annotate(
        f"Peak Model: Iter {p_it}\nSuccess: {p_sc:.1f}%\nClosure: {p_cl:.1f}%\nRetention: {p_rt:.1f}%",
        xy=(p_it, p_sc),
        xytext=(p_it + 400, p_sc + 4),
        arrowprops=dict(facecolor="#D84315", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(facecolor="#FFEBEE", edgecolor="#D84315", boxstyle="round,pad=0.4"),
        fontsize=9,
        fontweight="bold"
    )

ax1.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
ax1.set_ylim(-2, 102)
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Target line
ax1.axhline(y=70, color="#2E7D32", linestyle="--", alpha=0.5, linewidth=1.2)
ax1.text(b_iters[0] + 50, 71.5, "User Stopping Threshold: 70% Success", color="#2E7D32", fontsize=8.5, fontweight="bold")

# --- Panel 2: Continuous Rollout Training Telemetry ---
ax2 = axes[1]
ax2.set_facecolor(bg_color)
ax2.grid(True, linestyle="--", alpha=0.6, color=grid_color)
ax2.set_title("2. Online Rollout Training Dynamics (Rolling 100-Episode Window)", fontsize=13, fontweight="bold", pad=10)

if r_iters:
    r_succ_s = smooth(r_succ)
    r_drop_s = smooth(r_drop)
    r_clos_s = smooth(r_clos)
    r_ret_s = smooth(r_ret)
    
    ax2.plot(r_iters, r_succ, color="#81C784", alpha=0.3, linewidth=0.8)
    ax2.plot(r_iters, r_succ_s, color="#2E7D32", linewidth=2.2, label="Rollout Success Rate (Smoothed)")
    
    ax2.plot(r_iters, r_drop, color="#E57373", alpha=0.3, linewidth=0.8)
    ax2.plot(r_iters, r_drop_s, color="#D32F2F", linewidth=2.2, label="Rollout Drop Rate (Smoothed)")
    
    ax2.plot(r_iters, r_clos_s, color="#7B1FA2", linewidth=1.8, linestyle="-.", label="Rotational Closure (Smoothed)")
    ax2.plot(r_iters, r_ret_s, color="#1976D2", linewidth=1.8, linestyle=":", label="Object Retention (Smoothed)")
    
    ax2.axvline(x=opt_a_iter, color="#E65100", linestyle=":", linewidth=2.0)

ax2.set_xlabel("Iteration", fontsize=11, fontweight="bold")
ax2.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
ax2.set_ylim(-2, 102)
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Formatting X axis
for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))

# Overall Title
start_str = b_times[0].strftime("%H:%M") if b_times else "08:31"
end_str = b_times[-1].strftime("%H:%M") if b_times else "12:20"
fig.suptitle(f"Delto Hand In-Hand Manipulation — Training Run Telemetry (Today {start_str} to {end_str})\nOption A: 10% Error Tolerance, 70k Cap", fontsize=14, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()
print(f"Graph successfully saved to: {OUTPUT_PNG}")

