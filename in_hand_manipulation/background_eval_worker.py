#!/usr/bin/env python3
"""
background_eval_worker.py
=============================================================================
Continuous Asynchronous Background Evaluator for In-Hand Manipulation.

Monitors output directory for periodic milestone checkpoints (e.g. ppo_checkpoint_iter_*.pth),
evaluates each candidate on 500 randomized out-of-distribution episodes across all objects,
promotes the model that achieves the highest benchmark evaluation score,
deletes intermediate checkpoints to conserve disk space,
and synchronizes with the master curriculum pipeline upon stage completion.
=============================================================================
"""

import os
import sys
import time
import glob
import re
import shutil
import argparse
import numpy as np
import scipy.spatial.transform as transform
import torch
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo import DeltoInHandManipulationEnv, ActorNetwork

# Global worker state for multi-process pool
_worker_actor = None
_worker_env = None
_worker_epsilon_ratio = 0.10
_worker_smooth_alpha = 1.0


def _init_eval_worker(
    state_dict,
    obs_mean,
    obs_std,
    scene_xml_path,
    max_goal_angle,
    min_goal_angle,
    epsilon_ratio,
    smooth_alpha=0.70,
    sim2real_dr=False,
    action_delay_max=1,
    vision_delay_max=2,
    obs_noise_scale=0.80,
    dr_ratio=0.30,
):
    global _worker_actor, _worker_env, _worker_epsilon_ratio, _worker_smooth_alpha
    torch.set_num_threads(1)
    _worker_actor = ActorNetwork(obs_dim=39, act_dim=12)
    _worker_actor.load_state_dict(state_dict, strict=False)
    _worker_actor.eval()
    _worker_epsilon_ratio = epsilon_ratio
    _worker_smooth_alpha = smooth_alpha
    
    _worker_env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=1.5708,
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
        sim2real_dr=sim2real_dr,
        action_delay_max=action_delay_max,
        vision_delay_max=vision_delay_max,
        smooth_alpha=smooth_alpha,
        obs_noise_scale=obs_noise_scale,
    )


def _eval_single_episode(task_tuple):
    global _worker_actor, _worker_env, _worker_epsilon_ratio, _worker_smooth_alpha
    seed, range_id, obj_id, min_rad, max_rad = task_tuple
    np.random.seed(seed)
    obs, _ = _worker_env.reset(object_id=obj_id)
    
    if obj_id == 1:
        # Cylinder: dominant longitudinal spin with bounded tilt/nutation
        axis = np.array([np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), np.random.choice([-1.0, 1.0])])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-8)

    target_angle = float(np.random.uniform(min_rad, max_rad))
    rand_rot = transform.Rotation.from_rotvec(axis * target_angle)
    q_xyzw = rand_rot.as_quat()
    _worker_env.target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    
    obj_idx = _worker_env.model.jnt_qposadr[_worker_env.model.joint(f"obj_{obj_id}_joint").id]
    cur_obj_quat = _worker_env.data.qpos[obj_idx+3:obj_idx+7]
    _worker_env.prev_d_theta = _worker_env.compute_angular_distance(cur_obj_quat, _worker_env.target_quat)
    _worker_env.initial_d_theta = float(_worker_env.prev_d_theta)
    
    init_angle_deg = float(np.degrees(_worker_env.initial_d_theta))
    theta_rad = float(_worker_env.initial_d_theta)
    quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad ** 2))
    _worker_env.current_max_steps = min(800, quad_steps)
    _worker_env.set_epsilon_ratio(_worker_epsilon_ratio)
    _worker_env.set_epsilon(_worker_epsilon_ratio * theta_rad)
    
    # Refresh observation with target_quat and active DR noise
    obs = _worker_env._get_observation()
    
    step = 0
    terminated = False
    truncated = False
    step_info = {}
    goals_achieved = 0

    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            raw_act = _worker_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _worker_env.step(raw_act)
        goals_achieved = step_info.get("successes_in_episode", 0)
        if goals_achieved >= 1:
            break
            
    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
    raw_dropped = bool(step_info.get("raw_dropped", False))
    tol_thresh_deg = _worker_epsilon_ratio * init_angle_deg
    is_success = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)
    
    dropped_before_success = bool(raw_dropped and (goals_achieved == 0))
    pre_goal_retained = not dropped_before_success
    
    if goals_achieved > 0:
        closure = 1.0
    elif dropped_before_success:
        closure = 0.0
    else:
        closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))
        
    return {
        "range_id": range_id,
        "object_id": obj_id,
        "success": is_success,
        "pre_goal_retained": pre_goal_retained,
        "dropped_before_success": dropped_before_success,
        "raw_dropped": raw_dropped,
        "closure": closure,
        "goals": goals_achieved,
        "steps": step,
        "final_err_deg": final_err_deg,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    num_episodes: int,
    num_workers: int,
    scene_xml_path: str,
    max_goal_angle: float,
    min_goal_angle: float,
    epsilon_ratio: float,
    smooth_alpha: float = 0.70,
    sim2real_dr: bool = False,
    action_delay_max: int = 1,
    vision_delay_max: int = 2,
    obs_noise_scale: float = 0.80,
    dr_ratio: float = 0.30,
):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    init_args = (
        state_dict,
        obs_mean,
        obs_std,
        scene_xml_path,
        max_goal_angle,
        min_goal_angle,
        epsilon_ratio,
        smooth_alpha,
        sim2real_dr,
        action_delay_max,
        vision_delay_max,
        obs_noise_scale,
        dr_ratio,
    )
    
    # Construct balanced evaluation tasks:
    # Range 1: 10° to 40° (0.17453 to 0.69813 rad)
    # Range 2: 40° to 90° (0.69813 to 1.57080 rad)
    # Balanced across 3 objects: Cylinder (1), Cube (2), Cuboid (3)
    range_configs = [
        (1, np.radians(10.0), np.radians(40.0)),
        (2, np.radians(40.0), np.radians(90.0)),
    ]
    tasks = []
    eps_per_bucket = max(1, num_episodes // 6)
    task_idx = 0
    for r_id, min_rad, max_rad in range_configs:
        for obj_id in [1, 2, 3]:
            for _ in range(eps_per_bucket):
                seed = 70000 + task_idx
                tasks.append((seed, r_id, obj_id, min_rad, max_rad))
                task_idx += 1
                
    # Randomize task order across objects and ranges
    np.random.shuffle(tasks)

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_eval_worker, initargs=init_args) as pool:
        results = pool.map(_eval_single_episode, tasks)
        
    n = len(results)
    r1_res = [r for r in results if r["range_id"] == 1]
    r2_res = [r for r in results if r["range_id"] == 2]
    
    r1_succ = sum(1 for r in r1_res if r["success"]) / max(1, len(r1_res))
    r1_clos = float(np.mean([r["closure"] for r in r1_res])) if r1_res else 0.0
    r1_ret = sum(1 for r in r1_res if r["pre_goal_retained"]) / max(1, len(r1_res))
    
    r2_succ = sum(1 for r in r2_res if r["success"]) / max(1, len(r2_res))
    r2_clos = float(np.mean([r["closure"] for r in r2_res])) if r2_res else 0.0
    r2_ret = sum(1 for r in r2_res if r["pre_goal_retained"]) / max(1, len(r2_res))
    
    overall_succ = sum(1 for r in results if r["success"]) / n
    overall_clos = float(np.mean([r["closure"] for r in results]))
    overall_ret = sum(1 for r in results if r["pre_goal_retained"]) / n
    overall_drop = 1.0 - overall_ret
    raw_drop_rate = sum(1 for r in results if r["raw_dropped"]) / n
    
    # Composite score prioritizing overall excellence across both angle ranges:
    # 50% Success + 30% Closure + 20% Retention
    score = (0.50 * overall_succ) + (0.30 * overall_clos) + (0.20 * overall_ret)
    emi = overall_ret * overall_clos
    
    return {
        "success_rate": overall_succ,
        "mean_closure": overall_clos,
        "pre_goal_retention": overall_ret,
        "pre_goal_drop_rate": overall_drop,
        "raw_drop_rate": raw_drop_rate,
        "score": score,
        "emi": emi,
        "r1_succ": r1_succ,
        "r1_clos": r1_clos,
        "r1_ret": r1_ret,
        "r2_succ": r2_succ,
        "r2_clos": r2_clos,
        "r2_ret": r2_ret,
        "episodes": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Continuous Background Model Evaluator")
    parser.add_argument("--watch_dir", type=str, required=True, help="Directory to monitor for checkpoints")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of random test episodes per checkpoint (default: 500)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel evaluation workers (default: 4)")
    parser.add_argument("--max_goal_angle", type=float, default=1.57, help="Max target rotation angle in radians")
    parser.add_argument("--min_goal_angle", type=float, default=0.15, help="Min target rotation angle in radians")
    parser.add_argument("--epsilon_ratio", type=float, default=0.10, help="Target relative error threshold (default: 0.10)")
    parser.add_argument("--smooth_alpha", type=float, default=0.70, help="Action smoothing alpha (default: 0.70)")
    parser.add_argument("--delete_checkpoints", action="store_true", default=False, help="Delete intermediate milestone checkpoints after evaluation")
    parser.add_argument("--stop_flag", type=str, default="", help="Stop flag file path indicating training has completed")
    parser.add_argument("--best_output", type=str, default="", help="Destination path for best actor checkpoint")
    parser.add_argument("--update_global_best", action="store_true", help="Also promote to global best_model/best_model.pth")
    parser.add_argument("--sim2real_dr", action="store_true", help="Enable sim-to-real domain randomization during evaluation")
    parser.add_argument("--action_delay_max", type=int, default=1, help="Max action delay steps for DR evaluation (default: 1 -> 20ms)")
    parser.add_argument("--vision_delay_max", type=int, default=2, help="Max ArUco vision latency steps (default: 2 -> 40ms)")
    parser.add_argument("--obs_noise_scale", type=float, default=0.80, help="Observation noise scale factor (default: 0.80)")
    parser.add_argument("--dr_ratio", type=float, default=0.30, help="Domain randomization ratio (default: 0.30)")
    args = parser.parse_args()
    
    watch_dir = os.path.abspath(args.watch_dir)
    stop_flag = args.stop_flag if args.stop_flag else os.path.join(watch_dir, "training_done.flag")
    best_output = args.best_output if args.best_output else os.path.join(watch_dir, "ppo_best_actor.pth")
    eval_log = os.path.join(watch_dir, "eval_benchmark_history.csv")
    
    if not os.path.exists(eval_log):
        with open(eval_log, "w") as f:
            f.write("iteration,epsilon_ratio,success_rate,closure_rate,drop_rate,retention_rate,raw_drop_rate,r1_succ,r1_clos,r2_succ,r2_clos,composite_score,elapsed_sec,timestamp\n")
            
    print(f"\n[Background Evaluator] Started monitoring: {watch_dir}")
    print(f"  Evaluation Episodes : {args.num_episodes} (across Cylinder, Cube, Cuboid)")
    print(f"  Workers             : {args.num_workers}")
    print(f"  Target Tolerance    : {args.epsilon_ratio*100:.1f}% error | Max Angle: {np.degrees(args.max_goal_angle):.1f}°")
    print(f"  Sim-to-Real DR      : {'ACTIVE (ActDelay=' + str(args.action_delay_max*20) + 'ms, VisDelay=' + str(args.vision_delay_max*20) + 'ms, Noise=' + str(args.obs_noise_scale) + ', DR=±' + str(int(args.dr_ratio*100)) + '%)' if args.sim2real_dr else 'CLEAN SIM'}")
    print(f"  Best Model Output   : {best_output}")
    print(f"  Stop Flag Trigger   : {stop_flag}\n", flush=True)
    
    best_score = -1.0
    best_iter = -1
    
    # If a best model already exists, initialize baseline from local or global champion
    target_baseline = best_output
    if not os.path.exists(target_baseline) and args.update_global_best:
        global_champ = os.path.join(PROJECT_ROOT, "best_model", "best_model.pth")
        if os.path.exists(global_champ):
            target_baseline = global_champ
            
    if os.path.exists(target_baseline):
        try:
            print(f"[Background Evaluator] Benchmarking reigning champion baseline: {target_baseline}...")
            t0 = time.time()
            base_metrics = evaluate_checkpoint(
                target_baseline, 
                num_episodes=args.num_episodes, 
                num_workers=args.num_workers,
                scene_xml_path=args.scene_xml,
                max_goal_angle=args.max_goal_angle,
                min_goal_angle=args.min_goal_angle,
                epsilon_ratio=args.epsilon_ratio,
                smooth_alpha=args.smooth_alpha,
                sim2real_dr=args.sim2real_dr,
                action_delay_max=args.action_delay_max,
                vision_delay_max=args.vision_delay_max,
                obs_noise_scale=args.obs_noise_scale,
                dr_ratio=args.dr_ratio,
            )
            best_score = base_metrics["score"]
        except Exception as e:
            print(f"  Could not evaluate initial baseline: {e}")
            
    processed_iters = set()
    if os.path.exists(eval_log):
        try:
            with open(eval_log, "r") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if parts and parts[0].isdigit():
                        processed_iters.add(int(parts[0]))
            if processed_iters:
                print(f"[Background Evaluator] Resumed with {len(processed_iters)} previously evaluated milestones: {sorted(list(processed_iters))}")
        except Exception:
            pass
            
    while True:
        pattern = os.path.join(watch_dir, "ppo_checkpoint_iter_*.pth")
        all_ckpts = glob.glob(pattern)
        
        # Parse iterations
        available = []
        for p in all_ckpts:
            m = re.search(r"ppo_checkpoint_iter_(\d+)\.pth", os.path.basename(p))
            if m:
                it = int(m.group(1))
                if it not in processed_iters:
                    available.append((it, p))
                    
        available.sort(key=lambda x: x[0])
        
        if available:
            # Process oldest checkpoint in queue
            it, ckpt_file = available[0]
            processed_iters.add(it)
            
            # Ensure file is completely flushed to disk
            time.sleep(1.0)
            file_size = 0
            for _ in range(5):
                cur_sz = os.path.getsize(ckpt_file) if os.path.exists(ckpt_file) else 0
                if cur_sz > 0 and cur_sz == file_size:
                    break
                file_size = cur_sz
                time.sleep(0.5)
                
            print(f"[Background Evaluator] === Evaluating Checkpoint Iteration {it:05d} ({args.num_episodes} episodes) ===", flush=True)
            t_start = time.time()
            try:
                # Maintain the explicit benchmark tolerance specified via CLI (--epsilon_ratio)
                active_eps = args.epsilon_ratio if args.epsilon_ratio is not None else 0.10

                metrics = evaluate_checkpoint(
                    ckpt_file,
                    num_episodes=args.num_episodes,
                    num_workers=args.num_workers,
                    scene_xml_path=args.scene_xml,
                    max_goal_angle=args.max_goal_angle,
                    min_goal_angle=args.min_goal_angle,
                    epsilon_ratio=active_eps,
                    smooth_alpha=args.smooth_alpha,
                    sim2real_dr=args.sim2real_dr,
                    action_delay_max=args.action_delay_max,
                    vision_delay_max=args.vision_delay_max,
                    obs_noise_scale=args.obs_noise_scale,
                    dr_ratio=args.dr_ratio,
                )
                t_eval = time.time() - t_start
                
                score = metrics["score"]
                succ = metrics["success_rate"] * 100
                clos = metrics["mean_closure"] * 100
                pre_drop = metrics["pre_goal_drop_rate"] * 100
                pre_ret = metrics["pre_goal_retention"] * 100
                raw_drop = metrics["raw_drop_rate"] * 100
                
                # Check for improvement or close contender
                promoted_str = ""
                is_close = (score >= (best_score - 0.015)) and (score <= best_score) and (best_score > 0)
                
                if is_close and os.path.exists(best_output):
                    print(f"\n  [CLOSE CONTENDER DETECTED] Iteration {it:05d} scored {score:.4f}, within {best_score - score:.4f} of Best Actor ({best_score:.4f})!", flush=True)
                    # Save candidate backup so it is never deleted prematurely
                    close_backup = os.path.join(watch_dir, f"ppo_close_contender_iter_{it}.pth")
                    shutil.copy2(ckpt_file, close_backup)
                    print(f"  [Safety Backup] Preserved close contender to: {os.path.basename(close_backup)}", flush=True)
                    
                    # Re-evaluate current best actor on fresh sample
                    print(f"  [Verification] Re-evaluating current Best Actor ({os.path.basename(best_output)}) on fresh {args.num_episodes}-episode benchmark...", flush=True)
                    try:
                        t_reeval = time.time()
                        reeval_metrics = evaluate_checkpoint(
                            best_output,
                            num_episodes=args.num_episodes,
                            num_workers=args.num_workers,
                            scene_xml_path=args.scene_xml,
                            max_goal_angle=args.max_goal_angle,
                            min_goal_angle=args.min_goal_angle,
                            epsilon_ratio=active_eps,
                            smooth_alpha=args.smooth_alpha,
                            sim2real_dr=args.sim2real_dr,
                            action_delay_max=args.action_delay_max,
                            obs_noise_scale=args.obs_noise_scale,
                            dr_ratio=args.dr_ratio,
                        )
                        reeval_score = reeval_metrics["score"]
                        print(f"  [Verification Result] Best Actor fresh score: {reeval_score:.4f} (Prior baseline: {best_score:.4f}) in {time.time()-t_reeval:.1f}s", flush=True)
                        
                        # If candidate score is strictly higher than re-evaluated best actor score:
                        if score > reeval_score:
                            print(f"  [PROMOTION TRIGGER] Candidate {it:05d} ({score:.4f}) BEATS re-evaluated Best Actor ({reeval_score:.4f})! Promoting candidate!", flush=True)
                            best_score = score
                            best_iter = it
                            shutil.copy2(ckpt_file, best_output)
                            champion_dir = os.path.join(PROJECT_ROOT, "best_model")
                            if args.update_global_best and os.path.exists(champion_dir):
                                try:
                                    champion_path = os.path.join(champion_dir, "best_model.pth")
                                    shutil.copy2(ckpt_file, champion_path)
                                    readme_path = os.path.join(champion_dir, "README.md")
                                    with open(readme_path, "w") as rf:
                                        rf.write("# Best Champion Model (Stratified In-Hand Manipulation)\n\n")
                                        rf.write(f"- **Checkpoint File**: `best_model.pth`\n")
                                        rf.write(f"- **Source**: Iteration {it} (Promoted via Close-Contender Re-evaluation)\n")
                                        rf.write(f"- **Target Precision Envelope**: 10.0% Relative Error Tolerance across 10° to 90° Envelope\n")
                                        rf.write(f"- **Overall Benchmark Score**: {score:.4f}\n")
                                        rf.write(f"- **Overall Success Rate (@10% Error)**: {succ:.1f}%\n")
                                        rf.write(f"- **Overall Rotational Closure**: {clos:.1f}%\n")
                                        rf.write(f"- **Overall Pre-Goal Retention**: {pre_ret:.1f}%\n\n")
                                        rf.write(f"### Stratified Angle Range Breakdown\n")
                                        rf.write(f"- **Range 1 (10° to 40° Small/Medium)**: {metrics['r1_succ']*100:.1f}% Success | {metrics['r1_clos']*100:.1f}% Closure | {metrics['r1_ret']*100:.1f}% Retention\n")
                                        rf.write(f"- **Range 2 (40° to 90° Large Gaiting)**: {metrics['r2_succ']*100:.1f}% Success | {metrics['r2_clos']*100:.1f}% Closure | {metrics['r2_ret']*100:.1f}% Retention\n")
                                        rf.write(f"- **Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                except Exception as ce:
                                    print(f"  [Champion Update Warning]: {ce}", flush=True)
                            promoted_str = "  *** [HEAD-TO-HEAD RE-EVALUATION PROMOTION] ***"
                        else:
                            print(f"  [Verification Decision] Best Actor retains lead ({reeval_score:.4f} >= {score:.4f}). Updating baseline.", flush=True)
                            best_score = reeval_score
                    except Exception as ree:
                        print(f"  [Re-evaluation Error]: {ree}", flush=True)

                elif score > best_score:
                    best_score = score
                    best_iter = it
                    shutil.copy2(ckpt_file, best_output)
                    # Automatically update global champion repository
                    champion_dir = os.path.join(PROJECT_ROOT, "best_model")
                    if args.update_global_best and os.path.exists(champion_dir):
                        try:
                            champion_path = os.path.join(champion_dir, "best_model.pth")
                            shutil.copy2(ckpt_file, champion_path)
                            readme_path = os.path.join(champion_dir, "README.md")
                            with open(readme_path, "w") as rf:
                                rf.write("# Best Champion Model (Stratified In-Hand Manipulation)\n\n")
                                rf.write(f"- **Checkpoint File**: `best_model.pth`\n")
                                rf.write(f"- **Source**: Iteration {it}\n")
                                rf.write(f"- **Target Precision Envelope**: {active_eps*100:.1f}% Relative Error Tolerance across 10° to 90° Envelope\n")
                                rf.write(f"- **Overall Benchmark Score**: {score:.4f}\n")
                                rf.write(f"- **Overall Success Rate (@{active_eps*100:.1f}% Error)**: {succ:.1f}%\n")
                                rf.write(f"- **Overall Rotational Closure**: {clos:.1f}%\n")
                                rf.write(f"- **Overall Pre-Goal Retention**: {pre_ret:.1f}%\n\n")
                                rf.write(f"### Stratified Angle Range Breakdown\n")
                                rf.write(f"- **Range 1 (10° to 40° Small/Medium)**: {metrics['r1_succ']*100:.1f}% Success | {metrics['r1_clos']*100:.1f}% Closure | {metrics['r1_ret']*100:.1f}% Retention\n")
                                rf.write(f"- **Range 2 (40° to 90° Large Gaiting)**: {metrics['r2_succ']*100:.1f}% Success | {metrics['r2_clos']*100:.1f}% Closure | {metrics['r2_ret']*100:.1f}% Retention\n")
                                rf.write(f"- **Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        except Exception as ce:
                            print(f"  [Champion Update Warning]: {ce}", flush=True)
                    promoted_str = "  *** [NEW BEST BENCHMARK MODEL PROMOTED] ***"
                    
                print(f"  [Iter {it:05d} | Stratified 10-90°] R1(10-40°): {metrics['r1_succ']*100:4.1f}% succ, {metrics['r1_clos']*100:4.1f}% clos | R2(40-90°): {metrics['r2_succ']*100:4.1f}% succ, {metrics['r2_clos']*100:4.1f}% clos | Overall: {succ:4.1f}% succ, {clos:4.1f}% clos, {pre_ret:4.1f}% ret | Score: {score:.4f} ({t_eval:.1f}s){promoted_str}", flush=True)
                
                # Append to history log
                with open(eval_log, "a") as f:
                    f.write(f"{it},{active_eps:.4f},{metrics['success_rate']:.4f},{metrics['mean_closure']:.4f},{metrics['pre_goal_drop_rate']:.4f},{metrics['pre_goal_retention']:.4f},{metrics['raw_drop_rate']:.4f},{metrics['r1_succ']:.4f},{metrics['r1_clos']:.4f},{metrics['r2_succ']:.4f},{metrics['r2_clos']:.4f},{score:.4f},{t_eval:.2f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
            except Exception as e:
                print(f"  [Error evaluating checkpoint {ckpt_file}]: {e}", flush=True)
                
            # Delete intermediate milestone checkpoint only if explicitly requested
            if args.delete_checkpoints:
                try:
                    if os.path.exists(ckpt_file):
                        os.remove(ckpt_file)
                        # print(f"  [Cleanup] Deleted intermediate milestone: {os.path.basename(ckpt_file)}")
                except Exception as e:
                    print(f"  [Cleanup Warning]: Could not remove {ckpt_file}: {e}")
                
        else:
            # Queue is empty. Check if training has signaled completion
            if os.path.exists(stop_flag):
                print(f"\n[Background Evaluator] Detected training completion flag: {stop_flag}")
                print(f"  All intermediate checkpoints processed! Best model achieved at Iteration {best_iter:05d} with benchmark score {best_score:.4f}")
                print(f"[Background Evaluator] Exiting cleanly.", flush=True)
                break
            time.sleep(3.0)


if __name__ == "__main__":
    main()
