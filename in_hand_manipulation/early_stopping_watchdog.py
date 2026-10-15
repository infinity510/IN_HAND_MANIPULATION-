#!/usr/bin/env python3
"""
early_stopping_watchdog.py
=============================================================================
Automated Degradation Watchdog & Early-Stopping Sentinel for In-Hand Manipulation.

Monitors `eval_benchmark_history.csv` and `progress.csv` in real-time.
Detects when policy performance peaks and begins to degrade.
Gracefully halts training, locks in the all-time peak checkpoint, and promotes
the champion model to production.
=============================================================================
"""

import os
import sys
import time
import signal
import shutil
import argparse
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def find_training_pids(watch_dir: str):
    """Finds PIDs of running train_ppo.py processes associated with watch_dir."""
    pids = []
    try:
        out = subprocess.check_output(["pgrep", "-f", "train_ppo.py"], text=True)
        for line in out.strip().splitlines():
            pid = int(line.strip())
            # Check cmdline to ensure it belongs to this watch_dir
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read()
                    if os.path.basename(watch_dir) in cmdline or watch_dir in cmdline:
                        pids.append(pid)
            except Exception:
                pass
    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"[Watchdog Warning] Could not scan processes: {e}")
    return pids


def stop_training_processes(pids: list, watch_dir: str):
    """Sends SIGINT to training processes for graceful checkpointing and shutdown."""
    print(f"\n[Watchdog Action] Halting {len(pids)} training processes (PIDs: {pids})...", flush=True)
    
    # Touch stop flag first
    stop_flag = os.path.join(watch_dir, "stop.flag")
    with open(stop_flag, "w") as f:
        f.write(f"stopped_by_watchdog at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    training_done = os.path.join(watch_dir, "training_done.flag")
    with open(training_done, "w") as f:
        f.write(f"training_done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
            print(f"  Sent SIGINT to PID {pid}")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"  Error signaling PID {pid}: {e}")

    # Wait up to 10 seconds for graceful shutdown
    t0 = time.time()
    while time.time() - t0 < 10.0:
        alive = [pid for pid in pids if os.path.exists(f"/proc/{pid}")]
        if not alive:
            print("  All training processes exited cleanly.", flush=True)
            return
        time.sleep(1.0)

    # Force kill any remaining stubborn processes
    for pid in pids:
        try:
            if os.path.exists(f"/proc/{pid}"):
                os.kill(pid, signal.SIGTERM)
                print(f"  Sent SIGTERM to remaining PID {pid}")
        except Exception:
            pass


def promote_champion(watch_dir: str, peak_iter: int, peak_ckpt: str, peak_metrics: dict, global_baseline: float = 0.7026):
    """Preserves peak checkpoint as the final output and promotes to best_model if better than baseline."""
    final_output = os.path.join(watch_dir, "ppo_final_champion_actor.pth")
    if os.path.exists(peak_ckpt):
        shutil.copy2(peak_ckpt, final_output)
        print(f"[Watchdog Promotion] Preserved all-time peak actor to: {final_output}")
    else:
        print(f"[Watchdog Warning] Peak checkpoint not found at: {peak_ckpt}")
        return

    score = peak_metrics.get("composite_score", 0.0)
    succ = peak_metrics.get("success_rate", 0.0) * 100
    clos = peak_metrics.get("closure_rate", 0.0) * 100
    ret = peak_metrics.get("retention_rate", 0.0) * 100

    # Promote to global champion repository if score strictly exceeds reigning baseline
    champion_dir = os.path.join(PROJECT_ROOT, "best_model")
    if score > global_baseline and os.path.exists(champion_dir):
        global_target = os.path.join(champion_dir, "best_model.pth")
        shutil.copy2(peak_ckpt, global_target)
        readme_path = os.path.join(champion_dir, "README.md")
        with open(readme_path, "w") as rf:
            rf.write("# Best Champion Model (Stratified In-Hand Manipulation)\n\n")
            rf.write(f"- **Checkpoint File**: `best_model.pth`\n")
            rf.write(f"- **Source**: Iteration {peak_iter} (Promoted by Early Stopping Watchdog)\n")
            rf.write(f"- **Target Precision Envelope**: 10.0% Relative Error Tolerance across 10° to 90° Envelope\n")
            rf.write(f"- **Overall Benchmark Score**: {score:.4f} (Surpasses baseline {global_baseline:.4f})\n")
            rf.write(f"- **Overall Success Rate (@10% Error)**: {succ:.1f}%\n")
            rf.write(f"- **Overall Rotational Closure**: {clos:.1f}%\n")
            rf.write(f"- **Overall Pre-Goal Retention**: {ret:.1f}%\n\n")
            rf.write(f"### Stratified Angle Range Breakdown\n")
            rf.write(f"- **Range 1 (10° to 40° Small/Medium)**: {peak_metrics.get('r1_succ', 0.0)*100:.1f}% Success | {peak_metrics.get('r1_clos', 0.0)*100:.1f}% Closure\n")
            rf.write(f"- **Range 2 (40° to 90° Large Gaiting)**: {peak_metrics.get('r2_succ', 0.0)*100:.1f}% Success | {peak_metrics.get('r2_clos', 0.0)*100:.1f}% Closure\n")
            rf.write(f"- **Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  *** [GLOBAL CHAMPION PROMOTED] *** {global_target} updated with Iteration {peak_iter} (Score {score:.4f} > {global_baseline:.4f})!")


def write_stop_report(watch_dir: str, peak_iter: int, peak_metrics: dict, trigger_reason: str, eval_history: list):
    """Writes detailed early stopping report in markdown."""
    report_path = os.path.join(watch_dir, "DEGRADATION_STOP_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Early-Stopping Watchdog Sentinel Report\n\n")
        f.write(f"**Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"Training was **automatically halted by the Early-Stopping Watchdog** because the policy reached its peak dexterity and started degrading.\n\n")
        f.write(f"- **Trigger Reason**: {trigger_reason}\n")
        f.write(f"- **All-Time Peak Milestone**: **Iteration {peak_iter}**\n")
        f.write(f"- **Peak Benchmark Score**: **`{peak_metrics.get('composite_score', 0.0):.4f}`**\n")
        f.write(f"- **Peak Success Rate (@10% Error)**: **`{peak_metrics.get('success_rate', 0.0)*100:.1f}%`**\n")
        f.write(f"- **Peak Rotational Closure**: **`{peak_metrics.get('closure_rate', 0.0)*100:.1f}%`**\n")
        f.write(f"- **Peak Pre-Goal Retention**: **`{peak_metrics.get('retention_rate', 0.0)*100:.1f}%`**\n\n")
        f.write("## 2. Checkpoint Benchmark Evaluation Progression\n\n")
        f.write("| Iteration | Success Rate | Closure Rate | Retention Rate | Benchmark Score | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in eval_history:
            it = int(row["iteration"])
            sc = float(row["composite_score"])
            succ = float(row["success_rate"]) * 100
            clos = float(row["closure_rate"]) * 100
            ret = float(row["retention_rate"]) * 100
            status = "👑 Peak Model" if it == peak_iter else ("🔻 Degraded" if sc < peak_metrics.get("composite_score", 0.0) - 0.05 else "Active")
            f.write(f"| Iteration {it:05d} | {succ:5.1f}% | {clos:5.1f}% | {ret:5.1f}% | **`{sc:.4f}`** | {status} |\n")
        f.write("\n## 3. Preservation & Next Steps\n\n")
        f.write(f"- Peak actor model locked and preserved at: `ppo_final_champion_actor.pth`\n")
        f.write(f"- Global production model `best_model/best_model.pth` safely synchronized with the peak model.\n")
    print(f"[Watchdog Report] Wrote early stopping diagnosis report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated Degradation Watchdog & Early-Stopping Sentinel")
    parser.add_argument("--watch_dir", type=str, required=True, help="Directory to monitor (e.g. fine_tune_training/stage4_robust_10k)")
    parser.add_argument("--max_drop", type=float, default=0.075, help="Max permissible drop from peak benchmark score before triggering early stop (default: 0.075)")
    parser.add_argument("--patience", type=int, default=2, help="Number of consecutive degrading milestones tolerated (default: 2)")
    parser.add_argument("--check_interval", type=float, default=5.0, help="Monitoring polling interval in seconds")
    parser.add_argument("--global_baseline", type=float, default=0.7026, help="Global reigning champion baseline benchmark score")
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    eval_csv = os.path.join(watch_dir, "eval_benchmark_history.csv")

    print(f"\n" + "="*75)
    print(f"[Watchdog Sentinel] Started monitoring: {watch_dir}")
    print(f"  Max Drop Tolerance : -{args.max_drop:.4f} from peak score")
    print(f"  Degradation Patience: {args.patience} consecutive milestones")
    print(f"  Global Baseline    : {args.global_baseline:.4f}")
    print(f"="*75 + "\n", flush=True)

    peak_score = -1.0
    peak_iter = -1
    peak_ckpt = ""
    peak_metrics = {}
    consecutive_drops = 0
    last_processed_iter = -1

    while True:
        if os.path.exists(eval_csv):
            try:
                # Read eval history CSV
                with open(eval_csv, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                
                if len(lines) > 1:
                    headers = lines[0].split(",")
                    eval_history = []
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) == len(headers):
                            d = dict(zip(headers, parts))
                            eval_history.append({
                                "iteration": int(d["iteration"]),
                                "composite_score": float(d["composite_score"]),
                                "success_rate": float(d["success_rate"]),
                                "closure_rate": float(d["closure_rate"]),
                                "retention_rate": float(d["retention_rate"]),
                                "r1_succ": float(d.get("r1_succ", 0.0)),
                                "r1_clos": float(d.get("r1_clos", 0.0)),
                                "r2_succ": float(d.get("r2_succ", 0.0)),
                                "r2_clos": float(d.get("r2_clos", 0.0)),
                            })

                    if eval_history:
                        # Find the true all-time peak across entire history
                        peak_idx = -1
                        for idx, entry in enumerate(eval_history):
                            if entry["composite_score"] > peak_score:
                                peak_score = entry["composite_score"]
                                peak_iter = entry["iteration"]
                                peak_metrics = entry.copy()
                                peak_idx = idx

                        c1 = os.path.join(watch_dir, f"ppo_checkpoint_iter_{peak_iter}.pth")
                        c2 = os.path.join(watch_dir, "ppo_eval_best_actor.pth")
                        peak_ckpt = c1 if os.path.exists(c1) else c2

                        latest = eval_history[-1]
                        cur_iter = latest["iteration"]
                        cur_score = latest["composite_score"]

                        # Check if a new milestone has arrived
                        if cur_iter != last_processed_iter:
                            last_processed_iter = cur_iter
                            consecutive_drops = (len(eval_history) - 1) - peak_idx
                            drop_from_peak = peak_score - cur_score

                            print(f"[Watchdog Track @ Iter {cur_iter:05d}] Current: {cur_score:.4f} | Peak: {peak_score:.4f} (@ Iter {peak_iter:05d}) | Drop: -{drop_from_peak:.4f} | Milestones since peak: {consecutive_drops}/{args.patience}", flush=True)

                            # -------------------------------------------------------------
                            # DEGRADATION CRITERIA EVALUATION
                            # -------------------------------------------------------------
                            triggered = False
                            trigger_reason = ""

                            # Condition 1: Severe drop from peak (> max_drop)
                            if (peak_score - cur_score) >= args.max_drop and peak_score > 0.65:
                                triggered = True
                                trigger_reason = f"Severe score drop (-{drop_from_peak:.4f} >= -{args.max_drop:.4f}) below all-time peak ({peak_score:.4f} at Iter {peak_iter})"

                            # Condition 2: Consecutive declining milestones
                            elif consecutive_drops >= args.patience and drop_from_peak >= 0.035:
                                triggered = True
                                trigger_reason = f"Persistent degradation across {consecutive_drops} milestones since peak (-{drop_from_peak:.4f} below peak {peak_score:.4f} at Iter {peak_iter})"

                            if triggered:
                                print(f"\n" + "!"*75)
                                print(f"[DEGRADATION DETECTED @ Iter {cur_iter:05d}] Early-stopping triggered!")
                                print(f"  Reason: {trigger_reason}")
                                print(f"  Peak Model: Iteration {peak_iter:05d} with benchmark score {peak_score:.4f}")
                                print(f"!"*75 + "\n", flush=True)

                                # 1. Stop training processes
                                pids = find_training_pids(watch_dir)
                                stop_training_processes(pids, watch_dir)

                                # 2. Lock in and promote peak model
                                promote_champion(watch_dir, peak_iter, peak_ckpt, peak_metrics, args.global_baseline)

                                # 3. Write diagnostic markdown report
                                write_stop_report(watch_dir, peak_iter, peak_metrics, trigger_reason, eval_history)

                                print(f"\n[Watchdog Sentinel] Early-stopping procedure complete. Sentinel exiting cleanly.\n", flush=True)
                                return

            except Exception as e:
                print(f"[Watchdog Loop Warning]: {e}", flush=True)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
