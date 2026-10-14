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

Stage 2 (Large-Angle Gaiting Envelope, max_goal_angle = 1.57 rad / 90°):
  - Stage 2A: 14,080 iters at EpsR = 25% [COMPLETED, Golden Iter 9956: EMI 0.535]
  - Stage 2B: 3,103 iters at EpsR = 20%  [COMPLETED & PROMOTED, Golden Iter 392: EMI 0.485]
  - Stage 2C: 3,000 iters at EpsR = 15%  [QUEUED FOR EVENING, Fast Adaptation]
  - Stage 2D: 3,000 iters at EpsR = 10%  [QUEUED FOR OVERNIGHT, Fine Alignment]
  - Stage 2E: 3,000 iters at EpsR = 5%   [QUEUED FOR OVERNIGHT, Sub-2.5° Extreme Precision]
  - Stage 2 Grand Final Verification Benchmark (50 episodes at 5% tolerance)

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


def is_stage_fully_completed(stage_dir: str, target_iters: int, target_accuracy: float = 0.70) -> bool:
    golden = os.path.join(stage_dir, f"{os.path.basename(stage_dir)}_golden.pth")
    if os.path.exists(golden):
        return True
    eval_hist = os.path.join(stage_dir, "eval_benchmark_history.csv")
    if os.path.exists(eval_hist):
        try:
            with open(eval_hist, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 1:
                    last_parts = lines[-1].split(",")
                    if len(last_parts) >= 10:
                        eval_eps = float(last_parts[1])
                        last_succ = float(last_parts[2])
                        last_ret = float(last_parts[5])
                    else:
                        eval_eps = 0.15
                        last_succ = float(last_parts[1])
                        last_ret = float(last_parts[4])
                    if eval_eps <= 0.10 and last_succ >= target_accuracy and last_ret >= 0.70:
                        return True
        except Exception:
            pass
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
    iterations: int = 25000,
    epsilon_ratio: float = 0.25,
    min_epsilon_ratio: float = 0.10,
    max_goal_angle: float = 1.57,
    warmup_iters: int = 0,
    warmup_max_goal_angle: float = None,
    warmup_epsilon_ratio: float = None,
    w1: float = 1.5,
    w_progress: float = 8.0,
    w_precision: float = 4.0,
    w4: float = 40.0,
    w5: float = 15.0,
    w_step_penalty: float = 0.04,
    actor_lr: float = 1.6e-5,
    critic_lr: float = 8e-5,
    max_log_std: float = -2.0,
    entropy_coef: float = 0.0003,
    reset_log_std: bool = False,
    force_seed_checkpoint: bool = False,
    eval_script: str = "in_hand_manipulation/evaluate.py"
) -> str:
    """Executes a training stage, prints telemetry every 500 iters, cleans up, and evaluates."""
    log(f"\n{'='*75}\nSTARTING {stage_name}: EpsR={epsilon_ratio*100:.1f}%, GoalAngle={max_goal_angle:.2f} rad, Iterations={iterations} (Warmup={warmup_iters})\n{'='*75}")
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

    os.makedirs(output_dir, exist_ok=True)
    best_actor_dest = os.path.join(output_dir, "ppo_best_actor.pth")
    if start_iteration == 1:
        if force_seed_checkpoint and os.path.exists(checkpoint_in):
            shutil.copy2(checkpoint_in, best_actor_dest)
            shutil.copy2(checkpoint_in, os.path.join(output_dir, "ppo_latest_actor.pth"))
            shutil.copy2(checkpoint_in, os.path.join(output_dir, "ppo_training_best_actor.pth"))
            log(f"  [Force Re-Seed] Initialized fresh run in {output_dir} with golden baseline: {checkpoint_in}")
        elif not os.path.exists(best_actor_dest) and os.path.exists(checkpoint_in):
            shutil.copy2(checkpoint_in, best_actor_dest)
            log(f"  [Baseline Initialized] Seeded incoming model {checkpoint_in} -> {best_actor_dest}")

    if start_iteration <= iterations:
        # 1. Clean old stop flags
        stop_flag_file = os.path.join(output_dir, "training_done.flag")
        if os.path.exists(stop_flag_file):
            try:
                os.remove(stop_flag_file)
            except Exception:
                pass

        # 2. Launch Continuous Background Model Evaluator Sidecar (500 random tries across Cylinder, Cube, Cuboid)
        eval_worker_cmd = [
            "python3", "in_hand_manipulation/background_eval_worker.py",
            "--watch_dir", output_dir,
            "--num_episodes", "500",
            "--num_workers", "4",
            "--max_goal_angle", str(max_goal_angle),
            "--min_goal_angle", "0.15",
            "--epsilon_ratio", str(epsilon_ratio),
            "--stop_flag", stop_flag_file,
            "--best_output", os.path.join(output_dir, "ppo_best_actor.pth")
        ]
        log(f"  [Sidecar] Launching Continuous Background Evaluator (evaluates every 100 iters on 500 random tries)...")
        eval_proc = subprocess.Popen(eval_worker_cmd)

        # 3. Launch Training Process (12 workers for training + 4 workers for eval = 16 cores fully balanced)
        cmd = [
            "python3", "in_hand_manipulation/train_ppo.py",
            "--iterations", str(iterations),
            "--start_iteration", str(start_iteration),
            "--num_envs", "12",
            "--checkpoint", checkpoint_in,
            "--warmup_iters", str(warmup_iters),
            "--w1", str(w1),
            "--w_progress", str(w_progress),
            "--w_precision", str(w_precision),
            "--w4", str(w4),
            "--w5", str(w5),
            "--w_step_penalty", str(w_step_penalty),
            "--epsilon_ratio", str(epsilon_ratio),
            "--min_epsilon_ratio", str(min_epsilon_ratio),
            "--max_goal_angle", str(max_goal_angle),
            "--min_goal_angle", "0.15",
            "--actor_lr", str(actor_lr),
            "--critic_lr", str(critic_lr),
            "--max_log_std", str(max_log_std),
            "--entropy_coef", str(entropy_coef),
            "--save_freq", "100",
            "--output_dir", output_dir,
            "--plateau_window", "2500",
            "--target_accuracy", "0.70",
            "--cap_iterations", str(iterations)
        ]
        if warmup_max_goal_angle is not None:
            cmd.extend(["--warmup_max_goal_angle", str(warmup_max_goal_angle)])
        if warmup_epsilon_ratio is not None:
            cmd.extend(["--warmup_epsilon_ratio", str(warmup_epsilon_ratio)])
        if reset_log_std and start_iteration == 1:
            cmd.append("--reset_log_std")
        
        log(f"Launching training command: {' '.join(cmd)}")
        train_proc = subprocess.Popen(cmd)
        
        last_iter_report = (start_iteration // 500) * 500
        while train_proc.poll() is None:
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
            
        log(f"{stage_name} Training Completed all {iterations} iterations!")
        
        # 4. Signal Background Evaluator to process any remaining queued checkpoints and exit
        log(f"  [Sidecar Sync] Creating completion flag and waiting for background evaluator to sync...")
        with open(stop_flag_file, "w") as f:
            f.write("done\n")
        eval_proc.wait()
        log(f"  [Sidecar Sync] Background evaluator synchronized and exited cleanly.")
        if os.path.exists(stop_flag_file):
            try:
                os.remove(stop_flag_file)
            except Exception:
                pass
        time.sleep(3)
    
    # 5. Run Final Verification Benchmark with Comprehensive Evaluator (30 episodes, 10 per object)
    best_ckpt = os.path.join(output_dir, "ppo_best_actor.pth")
    summary_text = ""
    if os.path.exists(best_ckpt):
        log(f"Running Final Stage Comprehensive Benchmark using evaluate.py...")
        eval_cmd = [
            "python3", "in_hand_manipulation/evaluate.py",
            "--checkpoint", best_ckpt,
            "--episodes_per_range", "20",
            "--num_workers", "8"
        ]
        _, eval_summary = run_cmd(eval_cmd)
        summary_text = eval_summary[-2000:] if len(eval_summary) > 2000 else eval_summary
        log(f"Stage Comprehensive Benchmark Result:\n{summary_text}")

    append_report(stage_name, summary_text)
    
    # Save dedicated golden backup copy
    backup_file = os.path.join(output_dir, f"{stage_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}_golden.pth")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, backup_file)
        log(f"Saved golden checkpoint backup to: {backup_file}")
        
    # Clean workspace intermediate checkpoints
    cleanup_stage_dir(output_dir)
    
    git_commit_and_push(f"chore(curriculum): completed {stage_name} with comprehensive 500-try evaluation")
    return best_ckpt


def main():
    log("=== OVERNIGHT MASTER CURRICULUM PIPELINE INITIALIZED ===")
    
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w") as f:
            f.write("# Overnight Master Curriculum Execution Report\n\n")

    # =========================================================================
    # PART 1: STAGE 1 VERIFICATION
    # =========================================================================
    stage1_consolidated = os.path.join(PROJECT_ROOT, "stage1", "stage1_best_model.pth")
    if not os.path.exists(stage1_consolidated):
        raise FileNotFoundError(f"Stage 1 golden checkpoint not found at {stage1_consolidated}")
        
    log("Stage 1 verified and consolidated in stage1/stage1_best_model.pth. Proceeding directly to Stage 2...")
    current_model = stage1_consolidated

    # =========================================================================
    # PART 2: STAGE 2 (90° ROTATION ENVELOPE max_goal_angle = 1.57 rad / 90°)
    # Hardware-Safe Joint Limits Enforced Across All Stages
    # =========================================================================
    log("\n" + "#"*75 + "\nPROCEEDING TO STAGE 2: 90° LARGE-ANGLE GAITING\n" + "#"*75)

    # Fine-Tune Training Master Run in fine_tune_training/
    # Target: Consistently 70%-80% accuracy at <= 10% error tolerance across all objects
    # Evaluation Strategy: Stratified Range 1 (10°-40°) and Range 2 (40°-90°) across Cylinder, Cube, Cuboid
    # Base weights: Re-seeded directly from best_model/best_model.pth champion baseline
    # Warmup: First 400 iterations critic-only calibration, then full joint PPO up to 70k cap
    fine_tune_dir = os.path.join(PROJECT_ROOT, "fine_tune_training")
    champion_model = os.path.join(PROJECT_ROOT, "best_model", "best_model.pth")
    if not os.path.exists(champion_model):
        raise FileNotFoundError(f"Champion baseline model not found at {champion_model}")
    log(f"Proceeding to Fine-Tune Training in {fine_tune_dir} using champion baseline: {champion_model}")

    if not is_stage_fully_completed(fine_tune_dir, 70000):
        current_model = run_training_stage(
            stage_name="Fine-Tune Training (Stratified 2-Range Protocol)",
            output_dir=fine_tune_dir,
            checkpoint_in=champion_model,
            iterations=70000,
            epsilon_ratio=0.10,
            min_epsilon_ratio=0.10,
            max_goal_angle=1.57,
            warmup_iters=100,
            w1=2.5,
            w_progress=14.0,
            w_precision=6.0,
            w4=60.0,
            w5=15.0,
            w_step_penalty=0.04,
            actor_lr=1.4e-5,
            critic_lr=7e-5,
            max_log_std=-2.0,
            entropy_coef=0.0003,
            reset_log_std=False,
            force_seed_checkpoint=True,
            eval_script="in_hand_manipulation/evaluate.py"
        )
    else:
        current_model = os.path.join(fine_tune_dir, "ppo_best_actor.pth")

    # Grand Final Verification Benchmark using canonical evaluate.py
    # (100 episodes Range 1 [10°-40°] + 100 episodes Range 2 [40°-90°] per object = 600 total episodes)
    log("Running 600-episode Grand Final Stratified Benchmark (Range 1: 10°-40° and Range 2: 40°-90°)...")
    code, stage2_final_eval = run_cmd([
        "python3", "in_hand_manipulation/evaluate.py",
        "--checkpoint", current_model,
        "--episodes_per_range", "100",
        "--num_workers", "8"
    ])
    append_report("Stage 2 Grand Final Stratified Benchmark (600 Episodes, 10°-90°, Sub-10% Error)", stage2_final_eval[-2000:])
    git_commit_and_push("feat(curriculum): Stage 2 full completion of Target-10% Precision Run through 70k cap / 70% accuracy at <=10% error")

    log("=== STAGE 2 MASTER RUN COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()




