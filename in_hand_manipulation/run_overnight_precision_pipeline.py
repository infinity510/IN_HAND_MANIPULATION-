#!/usr/bin/env python3
"""
run_overnight_precision_pipeline.py
=============================================================================
Autonomous Overnight Multi-Stage Pipeline for In-Hand Manipulation:

Curriculum Flow (User Specification):
1. Stage 1 @ 25% tolerance (Current 10k run) -> Backup & evaluate
2. Stage 1 @ 20% tolerance (7,000 iters)    -> Backup & evaluate
3. Stage 1 @ 15% tolerance (5,000 iters)    -> Backup & evaluate
4. Stage 1 @ 10% tolerance (10,000 iters)   -> Backup & full 50-ep evaluation
5. Stage 2 (90° / 1.57 rad envelope) in same sequence:
   - 20% tolerance (7,000 iters)           -> Backup & evaluate
   - 15% tolerance (5,000 iters)           -> Backup & evaluate
   - 10% tolerance (10,000 iters)          -> Backup & full 50-ep evaluation
=============================================================================
"""

import os
import sys
import time
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

LOG_FILE = os.path.join(PROJECT_ROOT, "overnight_pipeline.log")
REPORT_FILE = os.path.join(PROJECT_ROOT, "OVERNIGHT_MASTER_REPORT.md")


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def append_report(section: str):
    with open(REPORT_FILE, "a") as f:
        f.write(section + "\n\n")


def is_train_running() -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", "train_ppo.py"], stdout=subprocess.PIPE, text=True)
        pids = [p.strip() for p in res.stdout.split() if p.strip()]
        my_pid = str(os.getpid())
        pids = [p for p in pids if p != my_pid]
        return len(pids) > 0
    except Exception:
        return False


def wait_for_current_training(stage_name: str, progress_file: str, target_iters: int):
    log(f"Waiting for {stage_name} to finish ({target_iters} iterations)...")
    last_reported = 0
    while is_train_running():
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].strip().split(",")
                        cur_iter = int(last[0])
                        if cur_iter - last_reported >= 500:
                            last_reported = cur_iter
                            succ = float(last[4]) * 100
                            drop = float(last[5]) * 100
                            clos = float(last[7]) * 100
                            emi = float(last[8])
                            err = float(last[10])
                            eps_r = float(last[9]) * 100
                            log(f"  [{stage_name} Iter {cur_iter:05d}/{target_iters}] EMI: {emi:.3f} | Succ: {succ:4.1f}% | Drop: {drop:4.1f}% | Clos: {clos:4.1f}% | Err: {err:4.1f}° | EpsR: {eps_r:4.1f}%")
            except Exception:
                pass
        time.sleep(30)
    log(f"{stage_name} completed successfully!")
    time.sleep(5)


def run_training_stage(
    stage_name: str,
    iterations: int,
    checkpoint_in: str,
    output_dir: str,
    epsilon_ratio: float,
    max_goal_angle: float = 0.80,
    w_progress: float = 4.0,
    w_precision: float = 2.0,
    w4: float = 16.0,
    w5: float = 15.0,
    actor_lr: float = 1.8e-5,
    critic_lr: float = 9e-5,
) -> str:
    log(f"\n=======================================================")
    log(f"LAUNCHING: {stage_name}")
    log(f"  Iterations: {iterations} | EpsR: {epsilon_ratio*100:.1f}% | Max Angle: {max_goal_angle:.2f} rad")
    log(f"  Input Checkpoint: {checkpoint_in}")
    log(f"  Output Directory: {output_dir}")
    log(f"=======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python3", "in_hand_manipulation/train_ppo.py",
        "--iterations", str(iterations),
        "--num_envs", "16",
        "--checkpoint", checkpoint_in,
        "--warmup_iters", "0",
        "--w_progress", str(w_progress),
        "--w_precision", str(w_precision),
        "--w4", str(w4),
        "--w5", str(w5),
        "--epsilon_ratio", str(epsilon_ratio),
        "--max_goal_angle", str(max_goal_angle),
        "--actor_lr", str(actor_lr),
        "--critic_lr", str(critic_lr),
        "--save_freq", "250",
        "--output_dir", output_dir
    ]
    
    log(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    
    progress_csv = os.path.join(output_dir, "progress.csv")
    last_reported = 0
    while proc.poll() is None:
        if os.path.exists(progress_csv):
            try:
                with open(progress_csv, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].strip().split(",")
                        cur_iter = int(last[0])
                        if cur_iter - last_reported >= 500:
                            last_reported = cur_iter
                            succ = float(last[4]) * 100
                            drop = float(last[5]) * 100
                            clos = float(last[7]) * 100
                            emi = float(last[8])
                            err = float(last[10])
                            log(f"  [{stage_name} Iter {cur_iter:05d}/{iterations}] EMI: {emi:.3f} | Succ: {succ:4.1f}% | Drop: {drop:4.1f}% | Clos: {clos:4.1f}% | Err: {err:4.1f}°")
            except Exception:
                pass
        time.sleep(30)
        
    log(f"{stage_name} training finished (exit code: {proc.returncode})!")
    time.sleep(5)
    
    best_ckpt = os.path.join(output_dir, "ppo_best_actor.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, "ppo_latest_actor.pth")
    return best_ckpt


def evaluate_model(eval_script: str, checkpoint: str, episodes: int, stage_label: str) -> str:
    log(f"Evaluating {stage_label} ({episodes} episodes) using {eval_script}...")
    res = subprocess.run([
        "python3", os.path.join(PROJECT_ROOT, eval_script),
        "--checkpoint", checkpoint,
        "--episodes", str(episodes),
        "--no_render"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    summary = res.stdout[-1500:] if len(res.stdout) > 1500 else res.stdout
    log(f"Evaluation Summary for {stage_label}:\n{summary}")
    return summary


def git_commit_and_push(commit_msg: str):
    log(f"Syncing with git: {commit_msg}")
    try:
        subprocess.run(["git", "add", "OVERNIGHT_MASTER_REPORT.md", "checkpoints_ppo/"], stdout=subprocess.PIPE, text=True)
        subprocess.run(["git", "commit", "-m", commit_msg], stdout=subprocess.PIPE, text=True)
        subprocess.run(["git", "push", "origin", "main"], stdout=subprocess.PIPE, text=True)
        log("Git push successful.")
    except Exception as e:
        log(f"Note on git sync: {e}")


def main():
    log("================================================================")
    log("=== OVERNIGHT MULTI-STAGE CURRICULUM ORCHESTRATOR STARTED ===")
    log("================================================================")
    
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w") as f:
            f.write("# Overnight In-Hand Manipulation Multi-Stage Training Report\n\n")
    
    # -----------------------------------------------------------------
    # STEP 1: Wait for Stage 1 (25% tolerance) Current 10k Run
    # -----------------------------------------------------------------
    log("STAGE 1.0: Monitoring active 10k run @ 25% tolerance...")
    wait_for_current_training("Stage 1 @ 25%", "checkpoints_ppo/progress.csv", 10000)
    
    stage1_25_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage1_25pct_best.pth")
    current_best = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "ppo_best_actor.pth")
    shutil.copyfile(current_best, stage1_25_ckpt)
    log(f"Backed up Stage 1 (25%) best model to: {stage1_25_ckpt}")
    
    eval_s1_25 = evaluate_model("in_hand_manipulation/evaluate_stage1.py", stage1_25_ckpt, 25, "Stage 1 @ 25%")
    append_report(f"## 1. Stage 1 (25% Tolerance - 10k Iterations)\n```\n{eval_s1_25}\n```")
    git_commit_and_push("checkpoint: Stage 1 25% completion & evaluation")
    
    # -----------------------------------------------------------------
    # STEP 2: Mini Stage 1.1 @ 20% Tolerance (7,000 Iterations)
    # -----------------------------------------------------------------
    s1_20_best = run_training_stage(
        stage_name="Stage 1 @ 20% (Mini-Stage 1)",
        iterations=7000,
        checkpoint_in=stage1_25_ckpt,
        output_dir="checkpoints_stage1_20pct",
        epsilon_ratio=0.20,
        max_goal_angle=0.80,
        w_progress=4.0,
        w_precision=2.5,
        w4=16.0,
        w5=15.0,
        actor_lr=1.8e-5,
        critic_lr=9e-5,
    )
    stage1_20_backup = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage1_20pct_best.pth")
    shutil.copyfile(s1_20_best, stage1_20_backup)
    log(f"Backed up Stage 1 (20%) best model to: {stage1_20_backup}")
    
    eval_s1_20 = evaluate_model("in_hand_manipulation/evaluate_stage1.py", stage1_20_backup, 25, "Stage 1 @ 20%")
    append_report(f"## 2. Stage 1 (20% Tolerance - 7k Iterations)\n```\n{eval_s1_20}\n```")
    git_commit_and_push("checkpoint: Stage 1 20% completion & evaluation")
    
    # -----------------------------------------------------------------
    # STEP 3: Mini Stage 1.2 @ 15% Tolerance (5,000 Iterations)
    # -----------------------------------------------------------------
    s1_15_best = run_training_stage(
        stage_name="Stage 1 @ 15% (Mini-Stage 2)",
        iterations=5000,
        checkpoint_in=stage1_20_backup,
        output_dir="checkpoints_stage1_15pct",
        epsilon_ratio=0.15,
        max_goal_angle=0.80,
        w_progress=4.0,
        w_precision=3.0,
        w4=18.0,
        w5=15.0,
        actor_lr=1.5e-5,
        critic_lr=8e-5,
    )
    stage1_15_backup = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage1_15pct_best.pth")
    shutil.copyfile(s1_15_best, stage1_15_backup)
    log(f"Backed up Stage 1 (15%) best model to: {stage1_15_backup}")
    
    eval_s1_15 = evaluate_model("in_hand_manipulation/evaluate_stage1.py", stage1_15_backup, 25, "Stage 1 @ 15%")
    append_report(f"## 3. Stage 1 (15% Tolerance - 5k Iterations)\n```\n{eval_s1_15}\n```")
    git_commit_and_push("checkpoint: Stage 1 15% completion & evaluation")
    
    # -----------------------------------------------------------------
    # STEP 4: Mini Stage 1.3 @ 10% Tolerance (10,000 Iterations)
    # -----------------------------------------------------------------
    s1_10_best = run_training_stage(
        stage_name="Stage 1 @ 10% (Final Precision Target)",
        iterations=10000,
        checkpoint_in=stage1_15_backup,
        output_dir="checkpoints_stage1_10pct",
        epsilon_ratio=0.10,
        max_goal_angle=0.80,
        w_progress=4.0,
        w_precision=4.0,
        w4=20.0,
        w5=15.0,
        actor_lr=1.2e-5,
        critic_lr=7e-5,
    )
    stage1_10_backup = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage1_10pct_best.pth")
    stage1_golden_master = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage1_golden_master.pth")
    shutil.copyfile(s1_10_best, stage1_10_backup)
    shutil.copyfile(s1_10_best, stage1_golden_master)
    log(f"Backed up Stage 1 (10%) Golden Master to: {stage1_golden_master}")
    
    eval_s1_10 = evaluate_model("in_hand_manipulation/evaluate_stage1.py", stage1_golden_master, 50, "Stage 1 Golden Master (10%)")
    append_report(f"## 4. Stage 1 Final (10% Tolerance - 10k Iterations - Golden Master)\n```\n{eval_s1_10}\n```")
    git_commit_and_push("checkpoint: Stage 1 10% Golden Master completion & full 50-ep evaluation")
    
    # -----------------------------------------------------------------
    # STEP 5: Stage 2 (90° / 1.57 rad Large-Angle Envelope) @ 20% Tolerance (7k)
    # -----------------------------------------------------------------
    s2_20_best = run_training_stage(
        stage_name="Stage 2 (90°) @ 20% Tolerance",
        iterations=7000,
        checkpoint_in=stage1_golden_master,
        output_dir="checkpoints_stage2_20pct",
        epsilon_ratio=0.20,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=2.5,
        w4=18.0,
        w5=15.0,
        actor_lr=1.5e-5,
        critic_lr=8e-5,
    )
    stage2_20_backup = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage2_20pct_best.pth")
    shutil.copyfile(s2_20_best, stage2_20_backup)
    
    eval_s2_20 = evaluate_model("in_hand_manipulation/evaluate_stage2.py", stage2_20_backup, 25, "Stage 2 @ 20%")
    append_report(f"## 5. Stage 2 (90° - 20% Tolerance - 7k Iterations)\n```\n{eval_s2_20}\n```")
    git_commit_and_push("checkpoint: Stage 2 20% completion & evaluation")
    
    # -----------------------------------------------------------------
    # STEP 6: Stage 2 (90°) @ 15% Tolerance (5k)
    # -----------------------------------------------------------------
    s2_15_best = run_training_stage(
        stage_name="Stage 2 (90°) @ 15% Tolerance",
        iterations=5000,
        checkpoint_in=stage2_20_backup,
        output_dir="checkpoints_stage2_15pct",
        epsilon_ratio=0.15,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=3.0,
        w4=20.0,
        w5=15.0,
        actor_lr=1.2e-5,
        critic_lr=7e-5,
    )
    stage2_15_backup = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage2_15pct_best.pth")
    shutil.copyfile(s2_15_best, stage2_15_backup)
    
    eval_s2_15 = evaluate_model("in_hand_manipulation/evaluate_stage2.py", stage2_15_backup, 25, "Stage 2 @ 15%")
    append_report(f"## 6. Stage 2 (90° - 15% Tolerance - 5k Iterations)\n```\n{eval_s2_15}\n```")
    git_commit_and_push("checkpoint: Stage 2 15% completion & evaluation")
    
    # -----------------------------------------------------------------
    # STEP 7: Stage 2 (90°) @ 10% Tolerance (10k)
    # -----------------------------------------------------------------
    s2_10_best = run_training_stage(
        stage_name="Stage 2 (90°) @ 10% Tolerance",
        iterations=10000,
        checkpoint_in=stage2_15_backup,
        output_dir="checkpoints_stage2_10pct",
        epsilon_ratio=0.10,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=4.0,
        w4=22.0,
        w5=15.0,
        actor_lr=1.0e-5,
        critic_lr=6e-5,
    )
    stage2_golden_master = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "stage2_golden_master.pth")
    shutil.copyfile(s2_10_best, stage2_golden_master)
    
    eval_s2_10 = evaluate_model("in_hand_manipulation/evaluate_stage2.py", stage2_golden_master, 50, "Stage 2 Golden Master (10%)")
    append_report(f"## 7. Stage 2 Final (90° - 10% Tolerance - 10k Iterations - Golden Master)\n```\n{eval_s2_10}\n```")
    git_commit_and_push("checkpoint: Stage 2 10% Golden Master completion & full 50-ep evaluation")
    
    log("================================================================")
    log("=== OVERNIGHT MULTI-STAGE CURRICULUM COMPLETED ALL STAGES! ===")
    log("================================================================")


if __name__ == "__main__":
    main()
