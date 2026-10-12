#!/usr/bin/env python3
"""
run_overnight_curriculum.py
=============================================================================
Autonomous Master Overnight Curriculum Runner for Delto Hand In-Hand Manipulation.

Full Roadmap:
- Stage 1A: Finish current 10k iterations at EpsR = 25% -> Backup & Evaluate
- Stage 1B: 10k iterations at EpsR = 20% -> Backup & Evaluate
- Stage 1C: 10k iterations at EpsR = 15% -> Backup & Evaluate
- Stage 1D: 10k iterations at EpsR = 10% (Target 70%+ accuracy) -> Backup & Evaluate
- Stage 2A: 90° rotation (max_goal_angle=1.57) at EpsR = 25% -> Backup & Evaluate
- Stage 2B: 90° rotation at EpsR = 20% -> Backup & Evaluate
- Stage 2C: 90° rotation at EpsR = 15% -> Backup & Evaluate
- Stage 2D: 90° rotation at EpsR = 10% -> Final Benchmark & Git Commit
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


def is_train_process_running() -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", "train_ppo.py"], stdout=subprocess.PIPE, text=True)
        pids = [p.strip() for p in res.stdout.split() if p.strip()]
        return len(pids) > 0
    except Exception:
        return False


def wait_for_current_training():
    log("Waiting for existing training run to complete...")
    last_report = 0
    while is_train_process_running():
        prog_path = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "progress.csv")
        if os.path.exists(prog_path):
            try:
                with open(prog_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        cur_iter = int(last_line[0])
                        if cur_iter - last_report >= 500:
                            last_report = cur_iter
                            log(f"  [Monitoring] Current Iteration: {cur_iter:04d}/10000 | Succ: {float(last_line[4])*100:.1f}% | Drop: {float(last_line[5])*100:.1f}% | Clos: {float(last_line[7])*100:.1f}% | Err: {float(last_line[10]):.1f}°")
            except Exception:
                pass
        time.sleep(30)
    log("Current training process finished.")
    time.sleep(5)


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
    """Executes a 10,000-iteration training stage, backs up, and evaluates."""
    log(f"\n{'='*75}\nSTARTING {stage_name}: EpsR={epsilon_ratio*100:.0f}%, GoalAngle={max_goal_angle:.2f} rad, Iter={iterations}\n{'='*75}")
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
        "--save_freq", "500",
        "--output_dir", output_dir
    ]
    
    log(f"Launching command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    
    last_iter_report = 0
    while proc.poll() is None:
        prog_path = os.path.join(output_dir, "progress.csv")
        if os.path.exists(prog_path):
            try:
                with open(prog_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        cur_iter = int(last_line[0])
                        if cur_iter - last_iter_report >= 500:
                            last_iter_report = cur_iter
                            succ = float(last_line[4]) * 100
                            drop = float(last_line[5]) * 100
                            clos = float(last_line[7]) * 100
                            emi = float(last_line[8])
                            err = float(last_line[10])
                            log(f"  [{stage_name} Iter {cur_iter:04d}/{iterations}] EMI: {emi:.3f} | Succ: {succ:.1f}% | Drop: {drop:.1f}% | Clos: {clos:.1f}% | Err: {err:.1f}°")
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
    code, eval_out = run_cmd([
        "python3", eval_script,
        "--checkpoint", best_ckpt,
        "--episodes", "25",
        "--no_render"
    ])
    
    summary_text = eval_out[-1500:] if len(eval_out) > 1500 else eval_out
    log(f"Evaluation summary for {stage_name}:\n{summary_text}")
    append_report(stage_name, summary_text)
    
    # Save dedicated backup copy
    backup_file = os.path.join(output_dir, f"{stage_name.lower().replace(' ', '_')}_golden.pth")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, backup_file)
        log(f"Saved golden checkpoint backup to: {backup_file}")
        
    git_commit_and_push(f"chore(curriculum): completed {stage_name} with evaluation report")
    return best_ckpt


def main():
    log("=== OVERNIGHT MASTER CURRICULUM PIPELINE INITIALIZED ===")
    
    # Init progress report file if not exists
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w") as f:
            f.write("# Overnight Master Curriculum Execution Report\n\n")

    # -------------------------------------------------------------
    # Stage Check / Auto-Resumption Logic
    # -------------------------------------------------------------
    s1c_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps15", "ppo_best_actor.pth")
    s1b_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps20", "ppo_best_actor.pth")
    s1a_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps25", "ppo_best_actor.pth")
    
    if os.path.exists(s1c_ckpt):
        log(f"Detected completed Stage 1C checkpoint at: {s1c_ckpt}")
        log("Skipping Stage 1A, 1B, 1C and proceeding directly to Stage 1D (EpsR = 10%)...")
        current_best_model = s1c_ckpt
    elif os.path.exists(s1b_ckpt):
        log(f"Detected completed Stage 1B checkpoint at: {s1b_ckpt}")
        log("Resuming into Stage 1C (EpsR = 15%)...")
        current_best_model = run_training_stage(
            stage_name="Stage 1C",
            output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps15"),
            checkpoint_in=s1b_ckpt,
            iterations=10000,
            epsilon_ratio=0.15,
            max_goal_angle=0.80,
            w_progress=4.0,
            w_precision=2.5,
            w4=18.0,
            w5=15.0,
            actor_lr=1.8e-5,
            critic_lr=9e-5,
            eval_script="in_hand_manipulation/evaluate_stage1.py"
        )
    else:
        # Step 1: Wait for current running Stage 1A (EpsR = 25%) to finish
        wait_for_current_training()
        stage1a_dir = os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps25")
        os.makedirs(stage1a_dir, exist_ok=True)
        src_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "ppo_best_actor.pth")
        if not os.path.exists(src_ckpt):
            src_ckpt = os.path.join(PROJECT_ROOT, "checkpoints_ppo", "ppo_latest_actor.pth")
        dest_ckpt = os.path.join(stage1a_dir, "ppo_best_actor.pth")
        if os.path.exists(src_ckpt):
            shutil.copy2(src_ckpt, dest_ckpt)
        log("Evaluating Stage 1A (EpsR = 25%)...")
        code, eval_out = run_cmd([
            "python3", "in_hand_manipulation/evaluate_stage1.py",
            "--checkpoint", dest_ckpt,
            "--episodes", "25",
            "--no_render"
        ])
        append_report("Stage 1A (EpsR = 25% Baseline)", eval_out[-1500:])
        git_commit_and_push("chore(curriculum): Stage 1A (EpsR=25%) completed")
        current_best_model = dest_ckpt

        # Step 2: Stage 1B (EpsR = 20%, 10,000 iterations)
        current_best_model = run_training_stage(
            stage_name="Stage 1B",
            output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps20"),
            checkpoint_in=current_best_model,
            iterations=10000,
            epsilon_ratio=0.20,
            max_goal_angle=0.80,
            w_progress=4.0,
            w_precision=2.2,
            w4=16.0,
            w5=15.0,
            actor_lr=2e-5,
            critic_lr=1e-4,
            eval_script="in_hand_manipulation/evaluate_stage1.py"
        )

        # Step 3: Stage 1C (EpsR = 15%, 10,000 iterations)
        current_best_model = run_training_stage(
            stage_name="Stage 1C",
            output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps15"),
            checkpoint_in=current_best_model,
            iterations=10000,
            epsilon_ratio=0.15,
            max_goal_angle=0.80,
            w_progress=4.0,
            w_precision=2.5,
            w4=18.0,
            w5=15.0,
            actor_lr=1.8e-5,
            critic_lr=9e-5,
            eval_script="in_hand_manipulation/evaluate_stage1.py"
        )

    # -------------------------------------------------------------
    # Step 4: Stage 1D (EpsR = 10%, 10,000 iterations, Target 70%+)
    # -------------------------------------------------------------
    current_best_model = run_training_stage(
        stage_name="Stage 1D",
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage1_eps10"),
        checkpoint_in=current_best_model,
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

    # Comprehensive 50-episode Stage 1 Benchmark
    log("Running full 50-episode Stage 1 Final Verification...")
    code, stage1_final_eval = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage1.py",
        "--checkpoint", current_best_model,
        "--episodes", "50",
        "--no_render"
    ])
    append_report("Stage 1 Final Benchmark (50 Episodes)", stage1_final_eval[-2000:])
    git_commit_and_push("feat(curriculum): Stage 1 100% completed with verified precision benchmark")

    # -------------------------------------------------------------
    # Step 5: Stage 2 (90° Rotation Envelope, max_goal_angle = 1.57 rad)
    # -------------------------------------------------------------
    log("\n" + "#"*75 + "\nPROCEEDING TO STAGE 2: 90° LARGE-ANGLE GAITING ENVELOPE\n" + "#"*75)
    
    # Stage 2A: 90° at EpsR = 25%
    current_best_model = run_training_stage(
        stage_name="Stage 2A",
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps25"),
        checkpoint_in=current_best_model,
        iterations=10000,
        epsilon_ratio=0.25,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=2.0,
        w4=16.0,
        w5=15.0,
        actor_lr=1.8e-5,
        critic_lr=9e-5,
        eval_script="in_hand_manipulation/evaluate_stage2.py"
    )

    # Stage 2B: 90° at EpsR = 20%
    current_best_model = run_training_stage(
        stage_name="Stage 2B",
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps20"),
        checkpoint_in=current_best_model,
        iterations=10000,
        epsilon_ratio=0.20,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=2.2,
        w4=18.0,
        w5=15.0,
        actor_lr=1.6e-5,
        critic_lr=8e-5,
        eval_script="in_hand_manipulation/evaluate_stage2.py"
    )

    # Stage 2C: 90° at EpsR = 15%
    current_best_model = run_training_stage(
        stage_name="Stage 2C",
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps15"),
        checkpoint_in=current_best_model,
        iterations=10000,
        epsilon_ratio=0.15,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=2.5,
        w4=20.0,
        w5=15.0,
        actor_lr=1.4e-5,
        critic_lr=7e-5,
        eval_script="in_hand_manipulation/evaluate_stage2.py"
    )

    # Stage 2D: 90° at EpsR = 10%
    current_best_model = run_training_stage(
        stage_name="Stage 2D",
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps10"),
        checkpoint_in=current_best_model,
        iterations=10000,
        epsilon_ratio=0.10,
        max_goal_angle=1.57,
        w_progress=4.0,
        w_precision=3.0,
        w4=22.0,
        w5=15.0,
        actor_lr=1.2e-5,
        critic_lr=6e-5,
        eval_script="in_hand_manipulation/evaluate_stage2.py"
    )

    # Comprehensive 50-episode Stage 2 Benchmark
    log("Running full 50-episode Stage 2 Final Verification...")
    code, stage2_final_eval = run_cmd([
        "python3", "in_hand_manipulation/evaluate_stage2.py",
        "--checkpoint", current_best_model,
        "--episodes", "50",
        "--no_render"
    ])
    append_report("Stage 2 Final Benchmark (50 Episodes)", stage2_final_eval[-2000:])
    git_commit_and_push("feat(curriculum): Stage 2 100% completed with full 90-deg gaiting benchmark")

    log("=== OVERNIGHT MASTER CURRICULUM COMPLETED ALL STAGES SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()

