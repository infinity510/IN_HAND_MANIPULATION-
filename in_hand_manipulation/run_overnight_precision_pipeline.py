#!/usr/bin/env python3
"""
run_overnight_precision_pipeline.py
=============================================================================
Autonomous overnight pipeline for Stage 1 In-Hand Manipulation:
1. Monitors Phase 1 training (10,000 iterations at fixed EpsR=25%).
2. Evaluates the resulting Phase 1 best model.
3. Automatically launches Phase 2: High-Precision Refinement (EpsR=10%, sub-5° error, 70%+ accuracy).
4. Evaluates the refined model with 50 test episodes.
5. Updates learning.md, commits, and pushes to git.
=============================================================================
"""

import os
import sys
import time
import subprocess
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

LOG_FILE = os.path.join(PROJECT_ROOT, "overnight_pipeline.log")


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_process_running(pattern: str) -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.PIPE, text=True)
        pids = [p.strip() for p in res.stdout.split() if p.strip()]
        # Ignore our own PID
        my_pid = str(os.getpid())
        pids = [p for p in pids if p != my_pid]
        return len(pids) > 0
    except Exception:
        return False


def run_cmd(cmd_list, timeout=None):
    log(f"Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return res.returncode, res.stdout


def main():
    log("=== OVERNIGHT AUTONOMOUS PIPELINE STARTED ===")
    
    # -------------------------------------------------------------
    # 1. MONITOR PHASE 1 (Currently running 10k training)
    # -------------------------------------------------------------
    log("Step 1: Monitoring Phase 1 10k training process...")
    last_iter_report = 0
    while is_process_running("train_ppo.py --iterations 10000"):
        # Check current iteration from progress.csv if available
        prog_path = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "progress.csv")
        if os.path.exists(prog_path):
            try:
                with open(prog_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        current_iter = int(last_line[0])
                        if current_iter - last_iter_report >= 500:
                            last_iter_report = current_iter
                            succ = float(last_line[4]) * 100
                            drop = float(last_line[5]) * 100
                            clos = float(last_line[7]) * 100
                            emi = float(last_line[8])
                            err = float(last_line[10])
                            log(f"  [Progress] Iter {current_iter:04d}/10000 | EMI: {emi:.3f} | Succ: {succ:.1f}% | Drop: {drop:.1f}% | Clos: {clos:.1f}% | Err: {err:.1f}°")
            except Exception:
                pass
        time.sleep(30)
    
    log("Phase 1 training has finished!")
    time.sleep(5)
    
    # -------------------------------------------------------------
    # 2. EVALUATE PHASE 1 MODEL
    # -------------------------------------------------------------
    phase1_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "ppo_best_actor.pth")
    if not os.path.exists(phase1_ckpt):
        phase1_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "ppo_latest_actor.pth")
    
    log(f"Step 2: Evaluating Phase 1 model: {phase1_ckpt}")
    code, eval_out = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage1.py",
        "--checkpoint", phase1_ckpt,
        "--episodes", "20",
        "--no_render"
    ])
    log("Phase 1 Evaluation Output:\n" + eval_out[-1200:])
    
    # -------------------------------------------------------------
    # 3. LAUNCH PHASE 2: PRECISION REFINEMENT (EpsR = 10%)
    # -------------------------------------------------------------
    prec_output_dir = os.path.join(PROJECT_ROOT, "checkpoints_ppo_precision")
    os.makedirs(prec_output_dir, exist_ok=True)
    
    log("Step 3: Launching Phase 2 High-Precision Training (EpsR=0.10, sub-5° target)...")
    phase2_cmd = [
        "python3", "in_hand_manipulation/train_ppo.py",
        "--iterations", "4000",
        "--num_envs", "16",
        "--checkpoint", phase1_ckpt,
        "--warmup_iters", "0",
        "--w_progress", "4.0",
        "--w_precision", "3.0",
        "--w4", "20.0",
        "--w5", "15.0",
        "--epsilon_ratio", "0.10",
        "--actor_lr", "1.5e-5",
        "--critic_lr", "8e-5",
        "--save_freq", "250",
        "--output_dir", "checkpoints_ppo_precision"
    ]
    
    log(f"Running Phase 2 Command: {' '.join(phase2_cmd)}")
    p2_proc = subprocess.Popen(phase2_cmd)
    
    p2_last_report = 0
    while p2_proc.poll() is None:
        p2_prog_path = os.path.join(prec_output_dir, "progress.csv")
        if os.path.exists(p2_prog_path):
            try:
                with open(p2_prog_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        current_iter = int(last_line[0])
                        if current_iter - p2_last_report >= 250:
                            p2_last_report = current_iter
                            succ = float(last_line[4]) * 100
                            drop = float(last_line[5]) * 100
                            clos = float(last_line[7]) * 100
                            emi = float(last_line[8])
                            err = float(last_line[10])
                            log(f"  [Precision Iter {current_iter:04d}/4000] EMI: {emi:.3f} | Succ (<=10%): {succ:.1f}% | Drop: {drop:.1f}% | Clos: {clos:.1f}% | Err: {err:.1f}°")
            except Exception:
                pass
        time.sleep(30)
    
    log("Phase 2 Precision Training completed!")
    time.sleep(5)
    
    # -------------------------------------------------------------
    # 4. FINAL VERIFICATION BENCHMARK (50 Episodes)
    # -------------------------------------------------------------
    final_ckpt = os.path.join(prec_output_dir, "ppo_best_actor.pth")
    if not os.path.exists(final_ckpt):
        final_ckpt = os.path.join(prec_output_dir, "ppo_latest_actor.pth")
        
    log(f"Step 4: Running 50-episode Final Verification Benchmark on: {final_ckpt}")
    code, final_eval_out = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage1.py",
        "--checkpoint", final_ckpt,
        "--episodes", "50",
        "--no_render"
    ])
    log("Final 50-Episode Benchmark Summary:\n" + final_eval_out[-1500:])
    
    # Save overnight summary report
    report_path = os.path.join(prec_output_dir, "OVERNIGHT_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Stage 1 Precision Refinement Overnight Report\n\n")
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n")
        f.write(final_eval_out[-2000:])
        f.write("\n```\n")
    log(f"Saved overnight report to: {report_path}")
    
    # -------------------------------------------------------------
    # 5. COMMIT & PUSH
    # -------------------------------------------------------------
    log("Step 5: Committing documentation and pushing to GitHub...")
    run_cmd(["git", "add", "learning.md", "checkpoints_ppo_precision/OVERNIGHT_REPORT.md"])
    run_cmd(["git", "commit", "-m", "chore: overnight stage 1 precision training and verification report"])
    run_cmd(["git", "push", "origin", "main"])
    
    log("=== OVERNIGHT PIPELINE FULLY COMPLETED! ===")


if __name__ == "__main__":
    main()

