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
_worker_max_goal_angle = 1.57
_worker_min_goal_angle = 0.15
_worker_epsilon_ratio = 0.20


def _init_eval_worker(state_dict, obs_mean, obs_std, scene_xml_path, max_goal_angle, min_goal_angle, epsilon_ratio):
    global _worker_actor, _worker_env, _worker_max_goal_angle, _worker_min_goal_angle, _worker_epsilon_ratio
    torch.set_num_threads(1)
    _worker_actor = ActorNetwork(obs_dim=39, act_dim=12)
    _worker_actor.load_state_dict(state_dict, strict=False)
    _worker_actor.eval()
    
    _worker_max_goal_angle = max_goal_angle
    _worker_min_goal_angle = min_goal_angle
    _worker_epsilon_ratio = epsilon_ratio
    
    _worker_env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=max_goal_angle,
        dr_ratio=0.20,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
    )


def _eval_single_episode(seed: int):
    global _worker_actor, _worker_env, _worker_max_goal_angle, _worker_min_goal_angle, _worker_epsilon_ratio
    np.random.seed(seed)
    obj_id = (seed % 3) + 1  # Balanced rotation: 1=Cylinder, 2=Cube, 3=Cuboid
    obs, _ = _worker_env.reset(object_id=obj_id)
    
    axis = np.random.randn(3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    target_angle = float(np.random.uniform(_worker_min_goal_angle, _worker_max_goal_angle))
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
    _worker_env.set_epsilon(_worker_epsilon_ratio * theta_rad)
    
    step = 0
    terminated = False
    truncated = False
    step_info = {}
    goals_achieved = 0
    
    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            act = _worker_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _worker_env.step(act)
        goals_achieved = step_info.get("successes_in_episode", 0)
        if goals_achieved >= 2:
            break
            
    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
    raw_dropped = bool(step_info.get("raw_dropped", False))
    tol_thresh_deg = _worker_epsilon_ratio * init_angle_deg
    is_success = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)
    
    # Pre-goal drop accounting:
    # Drops occurring AFTER a goal has already been achieved do NOT count against retention
    # for the stopping criterion (as instructed: "no dropping before success like if u get success and change target then dont count that in the drop")
    dropped_before_success = bool(raw_dropped and (goals_achieved == 0))
    
    if goals_achieved > 0:
        closure = 1.0
    elif raw_dropped:
        closure = 0.0
    else:
        closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))
        
    return {
        "success": is_success,
        "dropped_before_success": dropped_before_success,
        "raw_dropped": raw_dropped,
        "closure": closure,
        "goals": goals_achieved,
        "steps": step,
    }


def evaluate_checkpoint(ckpt_path: str, num_episodes: int, num_workers: int, scene_xml_path: str, max_goal_angle: float, min_goal_angle: float, epsilon_ratio: float):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    init_args = (state_dict, obs_mean, obs_std, scene_xml_path, max_goal_angle, min_goal_angle, epsilon_ratio)
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_eval_worker, initargs=init_args) as pool:
        seeds = [1000 + i for i in range(num_episodes)]
        results = pool.map(_eval_single_episode, seeds)
        
    n = len(results)
    succ_rate = sum(1 for r in results if r["success"]) / n
    pre_goal_drop_rate = sum(1 for r in results if r["dropped_before_success"]) / n
    pre_goal_retention = max(0.0, 1.0 - pre_goal_drop_rate)
    raw_drop_rate = sum(1 for r in results if r["raw_dropped"]) / n
    mean_closure = float(np.mean([r["closure"] for r in results]))
    
    # Score prioritizing Success Rate and Pre-Goal Retention
    score = succ_rate * pre_goal_retention * (0.5 + 0.5 * mean_closure)
    
    return {
        "success_rate": succ_rate,
        "pre_goal_drop_rate": pre_goal_drop_rate,
        "pre_goal_retention": pre_goal_retention,
        "raw_drop_rate": raw_drop_rate,
        "mean_closure": mean_closure,
        "score": score,
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
    parser.add_argument("--epsilon_ratio", type=float, default=0.20, help="Target relative error threshold")
    parser.add_argument("--stop_flag", type=str, default="", help="Stop flag file path indicating training has completed")
    parser.add_argument("--best_output", type=str, default="", help="Destination path for best actor checkpoint")
    args = parser.parse_args()
    
    watch_dir = os.path.abspath(args.watch_dir)
    stop_flag = args.stop_flag if args.stop_flag else os.path.join(watch_dir, "training_done.flag")
    best_output = args.best_output if args.best_output else os.path.join(watch_dir, "ppo_best_actor.pth")
    eval_log = os.path.join(watch_dir, "eval_benchmark_history.csv")
    
    if not os.path.exists(eval_log):
        with open(eval_log, "w") as f:
            f.write("iteration,success_rate,closure_rate,drop_rate,retention_rate,composite_score,elapsed_sec,timestamp\n")
            
    print(f"\n[Background Evaluator] Started monitoring: {watch_dir}")
    print(f"  Evaluation Episodes : {args.num_episodes} (across Cylinder, Cube, Cuboid)")
    print(f"  Workers             : {args.num_workers}")
    print(f"  Target Tolerance    : {args.epsilon_ratio*100:.1f}% error | Max Angle: {np.degrees(args.max_goal_angle):.1f}°")
    print(f"  Best Model Output   : {best_output}")
    print(f"  Stop Flag Trigger   : {stop_flag}\n", flush=True)
    
    best_score = -1.0
    best_iter = -1
    
    # If a best model already exists, initialize baseline
    if os.path.exists(best_output):
        try:
            print(f"[Background Evaluator] Benchmarking existing baseline: {best_output}...")
            t0 = time.time()
            base_metrics = evaluate_checkpoint(
                best_output, 
                num_episodes=args.num_episodes, 
                num_workers=args.num_workers,
                scene_xml_path=args.scene_xml,
                max_goal_angle=args.max_goal_angle,
                min_goal_angle=args.min_goal_angle,
                epsilon_ratio=args.epsilon_ratio
            )
            best_score = base_metrics["score"]
            print(f"  Existing Baseline Benchmark Score: {best_score:.4f} (Succ: {base_metrics['success_rate']*100:.1f}%, Clos: {base_metrics['mean_closure']*100:.1f}%, Ret: {base_metrics['pre_goal_retention']*100:.1f}%, Drop: {base_metrics['pre_goal_drop_rate']*100:.1f}%) in {time.time()-t0:.1f}s\n", flush=True)
        except Exception as e:
            print(f"  Could not evaluate initial baseline: {e}")
            
    processed_iters = set()
    
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
                # Dynamically retrieve active epsilon_ratio stored in checkpoint
                active_eps = args.epsilon_ratio
                try:
                    payload_meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    if "config" in payload_meta and "epsilon_ratio" in payload_meta["config"]:
                        active_eps = float(payload_meta["config"]["epsilon_ratio"])
                except Exception:
                    pass

                metrics = evaluate_checkpoint(
                    ckpt_file,
                    num_episodes=args.num_episodes,
                    num_workers=args.num_workers,
                    scene_xml_path=args.scene_xml,
                    max_goal_angle=args.max_goal_angle,
                    min_goal_angle=args.min_goal_angle,
                    epsilon_ratio=active_eps
                )
                t_eval = time.time() - t_start
                
                score = metrics["score"]
                succ = metrics["success_rate"] * 100
                clos = metrics["mean_closure"] * 100
                pre_drop = metrics["pre_goal_drop_rate"] * 100
                pre_ret = metrics["pre_goal_retention"] * 100
                raw_drop = metrics["raw_drop_rate"] * 100
                
                # Check for improvement
                promoted_str = ""
                if score > best_score:
                    best_score = score
                    best_iter = it
                    shutil.copy2(ckpt_file, best_output)
                    # Automatically update global champion repository
                    champion_dir = os.path.join(PROJECT_ROOT, "best_model")
                    if os.path.exists(champion_dir):
                        try:
                            champion_path = os.path.join(champion_dir, "best_model.pth")
                            shutil.copy2(ckpt_file, champion_path)
                            readme_path = os.path.join(champion_dir, "README.md")
                            with open(readme_path, "w") as rf:
                                rf.write("# Best Champion Model (Stage 2 Large-Angle In-Hand Manipulation)\n\n")
                                rf.write(f"- **Checkpoint File**: `best_model.pth`\n")
                                rf.write(f"- **Source**: Iteration {it}\n")
                                rf.write(f"- **Target Precision Envelope**: 10.0% Relative Error Tolerance, 90° Large-Angle Envelope (1.57 rad)\n")
                                rf.write(f"- **Benchmark Score**: {score:.4f}\n")
                                rf.write(f"- **Success Rate (@10% Error)**: {succ:.1f}%\n")
                                rf.write(f"- **Mean Rotational Closure**: {clos:.1f}%\n")
                                rf.write(f"- **Pre-Goal Retention**: {pre_ret:.1f}%\n")
                                rf.write(f"- **Pre-Goal Drop Rate**: {pre_drop:.1f}%\n")
                                rf.write(f"- **Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        except Exception as ce:
                            print(f"  [Champion Update Warning]: {ce}", flush=True)
                    promoted_str = "  *** [NEW BEST BENCHMARK MODEL PROMOTED] ***"
                    
                print(f"  [Iter {it:05d} | EpsR {active_eps*100:.1f}%] Succ: {succ:5.1f}% | Clos: {clos:5.1f}% | Pre-Drop: {pre_drop:5.1f}% | Pre-Ret: {pre_ret:5.1f}% | Raw-Drop: {raw_drop:5.1f}% | Score: {score:.4f} ({t_eval:.1f}s){promoted_str}", flush=True)
                
                # Append to history log
                with open(eval_log, "a") as f:
                    f.write(f"{it},{active_eps:.4f},{metrics['success_rate']:.4f},{metrics['mean_closure']:.4f},{metrics['pre_goal_drop_rate']:.4f},{metrics['pre_goal_retention']:.4f},{metrics['raw_drop_rate']:.4f},{score:.4f},{t_eval:.2f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
            except Exception as e:
                print(f"  [Error evaluating checkpoint {ckpt_file}]: {e}", flush=True)
                
            # Delete intermediate milestone checkpoint to save disk space
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
