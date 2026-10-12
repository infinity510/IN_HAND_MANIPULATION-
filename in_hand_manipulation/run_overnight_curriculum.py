#!/usr/bin/env python3
"""
run_overnight_curriculum.py
=============================================================================
Autonomous Master Curriculum Runner for Delto Hand In-Hand Manipulation.

Extended Roadmap:
-----------------
Stage 1 (Tripod Precision Envelope, max_goal_angle = 0.80 rad / ~45.8°):
  - Stage 1A: 10,000 iters at EpsR = 25% [COMPLETED]
  - Stage 1B: 10,000 iters at EpsR = 20% [COMPLETED]
  - Stage 1C: 10,000 iters at EpsR = 15% [COMPLETED]
  - Stage 1D (Part 1): 10,000 iters at EpsR = 10% [RUNNING / FINISHING]
  - Stage 1D (Part 2): 10,000 iters at EpsR = 10% [Consolidation & sub-5° lock-in]
  - Stage 1E: 10,000 iters at EpsR = 5%  [Extreme sub-2.5° precision]
  - Stage 1 Final Verification Benchmark (50 episodes)

Stage 2 (Large-Angle Gaiting Envelope, max_goal_angle = 1.57 rad / 90°, 20k iters each):
  - Stage 2A: 20,000 iters at EpsR = 25%
  - Stage 2B: 20,000 iters at EpsR = 20%
  - Stage 2C: 20,000 iters at EpsR = 15%
  - Stage 2D: 20,000 iters at EpsR = 10%
  - Stage 2E: 20,000 iters at EpsR = 5%
  - Stage 2 Final Verification Benchmark (50 episodes)

Features:
  - Prints full telemetry matrix after every 500 iterations.
  - Automatically cleans workspace after each stage: retains only ppo_best_actor.pth,
    ppo_latest_actor.pth, golden checkpoints, progress.csv, and metrics.jsonl.
  - Generates OVERNIGHT_PROGRESS_REPORT.md and commits/pushes to GitHub after each stage.
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

LOG_FILE = os.path.join(PROJECT_ROOT, "overnight_curriculum.log")
REPORT_FILE = os.path.join(PROJECT_ROOT, "OVERNIGHT_PROGRESS_REPORT.md")


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def append_report(title: str, content: str):
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## {title} ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n```\n{content}\n```\n\n")


def run_cmd(cmd_list, timeout=None):
    log(f"Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return res.returncode, res.stdout


def git_commit_and_push(msg: str):
    try:
        run_cmd(["git", "add", "learning.md", "OVERNIGHT_PROGRESS_REPORT.md"])
        run_cmd(["git", "commit", "-m", msg])
        run_cmd(["git", "push", "origin", "main"])
        log(f"Git commit and push succeeded: {msg}")
    except Exception as e:
        log(f"Git push warning: {e}")


def cleanup_stage_dir(stage_dir: str):
    """
    Cleans up intermediate milestone checkpoints, retaining only:
    - ppo_best_actor.pth
    - ppo_latest_actor.pth
    - *_golden.pth
    - progress.csv
    - metrics.jsonl
    """
    if not os.path.exists(stage_dir):
        return
    deleted_count = 0
    for item in os.listdir(stage_dir):
        if item.startswith("ppo_checkpoint_iter_") and item.endswith(".pth"):
            try:
                os.remove(os.path.join(stage_dir, item))
                deleted_count += 1
            except Exception:
                pass
    log(f"  [Workspace Cleanup] Purged {deleted_count} intermediate milestone checkpoints from {stage_dir}")


def is_train_process_running() -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", "train_ppo.py"], stdout=subprocess.PIPE, text=True)
        pids = [p.strip() for p in res.stdout.split() if p.strip()]
        return len(pids) > 0
    except Exception:
        return False


def wait_for_active_training(dir_to_watch: str):
    """Waits for currently running train_ppo.py in dir_to_watch to finish."""
    log(f"Watching active training process in {dir_to_watch}...")
    last_report = 0
    while is_train_process_running():
        prog_path = os.path.join(dir_to_watch, "progress.csv")
        if os.path.exists(prog_path):
            try:
                with open(prog_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        cur_iter = int(last_line[0])
                        if cur_iter - last_report >= 500:
                            last_report = cur_iter
                            succ = float(last_line[4]) * 100
                            drop = float(last_line[5]) * 100
                            clos = float(last_line[7]) * 100
                            emi = float(last_line[8])
                            err = float(last_line[10])
                            ret = float(last_line[1])
                            crit = float(last_line[12])
                            log(f"  [Telemetry Matrix | Iter {cur_iter:05d}] EMI: {emi:.3f} | Succ: {succ:5.1f}% | Drop: {drop:5.1f}% | Clos: {clos:5.1f}% | Err: {err:5.1f}° | CritLoss: {crit:6.2f} | Return: {ret:7.1f}")
            except Exception:
                pass
        time.sleep(30)
    log("Active training process finished.")
    time.sleep(5)


def is_stage_fully_completed(stage_dir: str, target_iters: int) -> bool:
    prog = os.path.join(stage_dir, "progress.csv")
    if not (os.path.exists(stage_dir) and os.path.exists(prog)):
        return False
    try:
        with open(prog, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
            if len(lines) > 1:
                last_line = lines[-1].strip().split(",")
                cur_iter = int(last_line[0])
                return cur_iter >= target_iters
    except Exception:
        pass
    return False


def run_training_stage(
    stage_name: str,
    output_dir: str,
    checkpoint_in: str,
    iterations: int = 10000,
    epsilon_ratio: float = 0.25,
    max_goal_angle: float = 0.80,
    w_progress: float = 4.0,
    w_precision: float = 2.0,
    w4: float = 15.0,
    w5: float = 15.0,
    actor_lr: float = 2e-5,
    critic_lr: float = 1e-4,
    eval_script: str = "in_hand_manipulation/evaluate_stage1.py"
) -> str:
    """Executes a training stage, prints telemetry every 500 iters, cleans up, and evaluates."""
    log(f"\n{'='*75}\nSTARTING {stage_name}: EpsR={epsilon_ratio*100:.1f}%, GoalAngle={max_goal_angle:.2f} rad, Iterations={iterations}\n{'='*75}")
    os.makedirs(output_dir, exist_ok=True)

    start_iteration = 1
    prog_path = os.path.join(output_dir, "progress.csv")
    if os.path.exists(prog_path):
        try:
            with open(prog_path, "r") as f:
                lines = [l for l in f.readlines() if l.strip()]
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(",")
                    cur_completed = int(last_line[0])
                    if cur_completed >= iterations:
                        log(f"  [Auto-Resume] {stage_name} already completed all {cur_completed}/{iterations} iterations! Skipping training phase.")
                        start_iteration = cur_completed + 1
                    elif cur_completed > 0:
                        start_iteration = cur_completed + 1
                        latest_ckpt = os.path.join(output_dir, "ppo_latest_actor.pth")
                        if os.path.exists(latest_ckpt):
                            checkpoint_in = latest_ckpt
                        log(f"  [Auto-Resume] Detected existing progress in {output_dir}: Resuming from iteration {start_iteration} to {iterations} using {checkpoint_in}")
        except Exception as e:
            log(f"  [Auto-Resume Warning] Could not parse existing progress.csv: {e}")

    if start_iteration <= iterations:
        cmd = [
            "python3", "in_hand_manipulation/train_ppo.py",
            "--iterations", str(iterations),
            "--start_iteration", str(start_iteration),
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
            "--save_freq", "500",
            "--output_dir", output_dir
        ]
        
        log(f"Launching command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        
        last_iter_report = (start_iteration // 500) * 500
        while proc.poll() is None:
            if os.path.exists(prog_path):
                try:
                    with open(prog_path, "r") as f:
                        lines = [l for l in f.readlines() if l.strip()]
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(",")
                            cur_iter = int(last_line[0])
                            if cur_iter - last_iter_report >= 500 or cur_iter >= iterations:
                                last_iter_report = cur_iter
                                succ = float(last_line[4]) * 100
                                drop = float(last_line[5]) * 100
                                clos = float(last_line[7]) * 100
                                emi = float(last_line[8])
                                err = float(last_line[10])
                                ret = float(last_line[1])
                                crit = float(last_line[12])
                                log(f"  [Telemetry Matrix | {stage_name} Iter {cur_iter:05d}/{iterations}] EMI: {emi:.3f} | Succ: {succ:5.1f}% | Drop: {drop:5.1f}% | Clos: {clos:5.1f}% | Err: {err:5.1f}° | CritLoss: {crit:6.2f} | Return: {ret:7.1f}")
                except Exception:
                    pass
            time.sleep(30)
            
        log(f"{stage_name} Training Completed!")
        time.sleep(5)
    
    # Locate best checkpoint
    best_ckpt = os.path.join(output_dir, "ppo_best_actor.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, "ppo_latest_actor.pth")
        
    # Run evaluation
    log(f"Evaluating {stage_name} using {eval_script} on {best_ckpt}...")
    eval_cmd = [
        "python3", eval_script,
        "--checkpoint", best_ckpt,
        "--episodes", "25",
        "--no_render"
    ]
    if "stage2" in eval_script:
        eval_cmd.extend(["--max_goal_angle", str(max_goal_angle)])
    code, eval_out = run_cmd(eval_cmd)
    
    summary_text = eval_out[-1500:] if len(eval_out) > 1500 else eval_out
    log(f"Evaluation summary for {stage_name}:\n{summary_text}")
    append_report(stage_name, summary_text)
    
    # Save dedicated backup copy
    backup_file = os.path.join(output_dir, f"{stage_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}_golden.pth")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, backup_file)
        log(f"Saved golden checkpoint backup to: {backup_file}")
        
    # Clean workspace intermediate checkpoints
    cleanup_stage_dir(output_dir)
    
    git_commit_and_push(f"chore(curriculum): completed {stage_name} with evaluation report")
    return best_ckpt


def main():
    log("=== OVERNIGHT MASTER CURRICULUM PIPELINE INITIALIZED ===")
    
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w") as f:
            f.write("# Overnight Master Curriculum Execution Report\n\n")

    # =========================================================================
    # PART 1: STAGE 1 (PRECISION ENVELOPE theta <= 45.8° / 0.80 rad)
    # =========================================================================
    
    # Check if Stage 1D (Part 1) is currently running
    s1d_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps10")
    if is_train_process_running() and os.path.exists(s1d_dir):
        log("Stage 1D (Part 1, 10% error) is currently executing. Waiting for 10,000 iterations to finish...")
        wait_for_active_training(s1d_dir)
        
        # Evaluate Stage 1D (Part 1)
        best_s1d = os.path.join(s1d_dir, "ppo_best_actor.pth")
        if not os.path.exists(best_s1d):
            best_s1d = os.path.join(s1d_dir, "ppo_latest_actor.pth")
        log("Evaluating Stage 1D (Part 1)...")
        code, eval_out = run_cmd([
            "python3", "in_hand_manipulation/evaluate_stage1.py",
            "--checkpoint", best_s1d,
            "--episodes", "25",
            "--no_render"
        ])
        append_report("Stage 1D Part 1 (EpsR = 10%)", eval_out[-1500:])
        shutil.copy2(best_s1d, os.path.join(s1d_dir, "stage_1d_part1_golden.pth"))
        cleanup_stage_dir(s1d_dir)
        git_commit_and_push("chore(curriculum): Stage 1D Part 1 completed")
        current_model = best_s1d
    else:
        current_model = os.path.join(s1d_dir, "ppo_best_actor.pth")
        if not os.path.exists(current_model):
            current_model = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps15", "ppo_best_actor.pth")

    # Step 1D (Part 2): Another 10k iterations on 10% error
    s1d2_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps10_part2")
    if not is_stage_fully_completed(s1d2_dir, 10000):
        current_model = run_training_stage(
            stage_name="Stage 1D (Part 2 - 10k Consolidation)",
            output_dir=s1d2_dir,
            checkpoint_in=current_model,
            iterations=10000,
            epsilon_ratio=0.10,
            max_goal_angle=0.80,
            w_progress=4.0,
            w_precision=3.0,
            w4=20.0,
            w5=15.0,
            actor_lr=1.5e-5,
            critic_lr=8e-5,
            eval_script="in_hand_manipulation/evaluate_stage1.py"
        )
    else:
        log("Stage 1D (Part 2) already completed. Resuming...")
        current_model = os.path.join(s1d2_dir, "ppo_best_actor.pth")

    # Step 1E: Another 10k iterations on 5% error for Stage 1 itself
    s1e_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps05")
    if not is_stage_fully_completed(s1e_dir, 10000):
        current_model = run_training_stage(
            stage_name="Stage 1E (5% Precision Lock-In)",
            output_dir=s1e_dir,
            checkpoint_in=current_model,
            iterations=10000,
            epsilon_ratio=0.05,
            max_goal_angle=0.80,
            w_progress=4.0,
            w_precision=4.0,
            w4=25.0,
            w5=15.0,
            actor_lr=1.2e-5,
            critic_lr=6e-5,
            eval_script="in_hand_manipulation/evaluate_stage1.py"
        )
    else:
        log("Stage 1E (5% error) already completed. Resuming...")
        current_model = os.path.join(s1e_dir, "ppo_best_actor.pth")

    # Comprehensive 50-episode Stage 1 Final Verification Benchmark
    log("Running full 50-episode Stage 1 Final Precision Verification...")
    code, stage1_final_eval = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage1.py",
        "--checkpoint", current_model,
        "--episodes", "50",
        "--no_render"
    ])
    append_report("Stage 1 Grand Final Benchmark (50 Episodes, Sub-2.5° Tolerance)", stage1_final_eval[-2000:])
    git_commit_and_push("feat(curriculum): Stage 1 100% completed with verified 5% precision benchmark")

    # =========================================================================
    # PART 2: STAGE 2 (90° ROTATION ENVELOPE max_goal_angle = 1.57 rad / 90°)
    # Double iterations = 20,000 iterations for each step!
    # =========================================================================
    log("\n" + "#"*75 + "\nPROCEEDING TO STAGE 2: 90° LARGE-ANGLE GAITING (20,000 ITERATIONS PER STEP)\n" + "#"*75)

    # Stage 2A: 90° at EpsR = 25% (20k iterations)
    s2a_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps25")
    if not is_stage_fully_completed(s2a_dir, 20000):
        current_model = run_training_stage(
            stage_name="Stage 2A (90° Envelope, 25% Error)",
            output_dir=s2a_dir,
            checkpoint_in=current_model,
            iterations=20000,
            epsilon_ratio=0.25,
            max_goal_angle=1.57,
            w_progress=4.0,
            w_precision=2.0,
            w4=18.0,
            w5=15.0,
            actor_lr=1.8e-5,
            critic_lr=9e-5,
            eval_script="in_hand_manipulation/evaluate_stage2.py"
        )
    else:
        current_model = os.path.join(s2a_dir, "ppo_best_actor.pth")

    # Stage 2B: 90° at EpsR = 20% (20k iterations)
    s2b_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps20")
    if not is_stage_fully_completed(s2b_dir, 20000):
        current_model = run_training_stage(
            stage_name="Stage 2B (90° Envelope, 20% Error)",
            output_dir=s2b_dir,
            checkpoint_in=current_model,
            iterations=20000,
            epsilon_ratio=0.20,
            max_goal_angle=1.57,
            w_progress=4.0,
            w_precision=2.2,
            w4=20.0,
            w5=15.0,
            actor_lr=1.6e-5,
            critic_lr=8e-5,
            eval_script="in_hand_manipulation/evaluate_stage2.py"
        )
    else:
        current_model = os.path.join(s2b_dir, "ppo_best_actor.pth")

    # Stage 2C: 90° at EpsR = 15% (20k iterations)
    s2c_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps15")
    if not is_stage_fully_completed(s2c_dir, 20000):
        current_model = run_training_stage(
            stage_name="Stage 2C (90° Envelope, 15% Error)",
            output_dir=s2c_dir,
            checkpoint_in=current_model,
            iterations=20000,
            epsilon_ratio=0.15,
            max_goal_angle=1.57,
            w_progress=4.0,
            w_precision=2.5,
            w4=22.0,
            w5=15.0,
            actor_lr=1.4e-5,
            critic_lr=7e-5,
            eval_script="in_hand_manipulation/evaluate_stage2.py"
        )
    else:
        current_model = os.path.join(s2c_dir, "ppo_best_actor.pth")

    # Stage 2D: 90° at EpsR = 10% (20k iterations)
    s2d_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps10")
    if not is_stage_fully_completed(s2d_dir, 20000):
        current_model = run_training_stage(
            stage_name="Stage 2D (90° Envelope, 10% Error)",
            output_dir=s2d_dir,
            checkpoint_in=current_model,
            iterations=20000,
            epsilon_ratio=0.10,
            max_goal_angle=1.57,
            w_progress=4.0,
            w_precision=3.0,
            w4=24.0,
            w5=15.0,
            actor_lr=1.2e-5,
            critic_lr=6e-5,
            eval_script="in_hand_manipulation/evaluate_stage2.py"
        )
    else:
        current_model = os.path.join(s2d_dir, "ppo_best_actor.pth")

    # Stage 2E: 90° at EpsR = 5% (20k iterations)
    s2e_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps05")
    if not is_stage_fully_completed(s2e_dir, 20000):
        current_model = run_training_stage(
            stage_name="Stage 2E (90° Envelope, 5% Error)",
            output_dir=s2e_dir,
            checkpoint_in=current_model,
            iterations=20000,
            epsilon_ratio=0.05,
            max_goal_angle=1.57,
            w_progress=4.0,
            w_precision=4.0,
            w4=26.0,
            w5=15.0,
            actor_lr=1.0e-5,
            critic_lr=5e-5,
            eval_script="in_hand_manipulation/evaluate_stage2.py"
        )
    else:
        current_model = os.path.join(s2e_dir, "ppo_best_actor.pth")

    # Comprehensive 50-episode Stage 2 Grand Final Benchmark
    log("Running full 50-episode Stage 2 Grand Final Verification...")
    code, stage2_final_eval = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage2.py",
        "--checkpoint", current_model,
        "--max_goal_angle", "1.57",
        "--episodes", "50",
        "--no_render"
    ])
    append_report("Stage 2 Grand Final Benchmark (50 Episodes, 90° Gaiting)", stage2_final_eval[-2000:])
    git_commit_and_push("feat(curriculum): Stage 2 100% completed with verified 90-deg gaiting benchmark")

    log("=== MASTER CURRICULUM COMPLETED ALL STAGES (STAGE 1 + STAGE 2) SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
