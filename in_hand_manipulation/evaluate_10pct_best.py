#!/usr/bin/env python3
"""
evaluate_10pct_best.py
=============================================================================
Evaluates the best model (and latest checkpoint) at 10% relative error tolerance
across all 3 objects (100 Cylinder, 100 Cube, 100 Cuboid = 300 test episodes total)
under the 90° large-angle rotation envelope.
=============================================================================
"""

import os
import sys
import time
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

# Multiprocessing global worker state
_worker_actor = None
_worker_env = None
_worker_max_goal_angle = 1.57
_worker_min_goal_angle = 0.15
_worker_epsilon_ratio = 0.10


def _init_bench_worker(state_dict, obs_mean, obs_std, scene_xml_path, max_goal_angle, min_goal_angle, epsilon_ratio):
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


def _eval_single_run(args_tuple):
    global _worker_actor, _worker_env, _worker_max_goal_angle, _worker_min_goal_angle, _worker_epsilon_ratio
    seed, obj_id = args_tuple
    np.random.seed(seed)
    
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
    
    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            act = _worker_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _worker_env.step(act)
        goals = step_info.get("successes_in_episode", 0)
        if goals >= 2:
            break
            
    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
    raw_dropped = bool(step_info.get("raw_dropped", False))
    goals_achieved = step_info.get("successes_in_episode", 0)
    
    # Pre-goal drop accounting
    dropped_before_success = bool(raw_dropped and (goals_achieved == 0))
    
    # Tolerances
    tol05_deg = 0.05 * init_angle_deg
    tol10_deg = 0.10 * init_angle_deg
    tol15_deg = 0.15 * init_angle_deg
    
    succ05 = bool(goals_achieved > 0 or final_err_deg <= tol05_deg)
    succ10 = bool(goals_achieved > 0 or final_err_deg <= tol10_deg)
    succ15 = bool(goals_achieved > 0 or final_err_deg <= tol15_deg)
    
    if goals_achieved > 0:
        closure = 1.0
    elif raw_dropped:
        closure = 0.0
    else:
        closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))
        
    return {
        "object_id": obj_id,
        "succ05": succ05,
        "succ10": succ10,
        "succ15": succ15,
        "dropped_before_success": dropped_before_success,
        "raw_dropped": raw_dropped,
        "closure": closure,
        "final_err_deg": final_err_deg,
        "init_angle_deg": init_angle_deg,
        "steps": step,
    }


def evaluate_model(ckpt_path: str, scene_xml_path: str, episodes_per_obj: int = 100, num_workers: int = 4):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    init_args = (state_dict, obs_mean, obs_std, scene_xml_path, 1.57, 0.15, 0.10)
    
    obj_names = {1: "Cylinder", 2: "Cube", 3: "Cuboid"}
    summary_by_obj = {}
    
    all_runs = []
    for obj_id in [1, 2, 3]:
        for ep in range(episodes_per_obj):
            seed = 42000 + (obj_id * 1000) + ep
            all_runs.append((seed, obj_id))
            
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_bench_worker, initargs=init_args) as pool:
        results = pool.map(_eval_single_run, all_runs)
        
    for obj_id in [1, 2, 3]:
        obj_res = [r for r in results if r["object_id"] == obj_id]
        n = len(obj_res)
        succ10_rate = sum(1 for r in obj_res if r["succ10"]) / n
        succ05_rate = sum(1 for r in obj_res if r["succ05"]) / n
        succ15_rate = sum(1 for r in obj_res if r["succ15"]) / n
        pre_drop_rate = sum(1 for r in obj_res if r["dropped_before_success"]) / n
        retention = 1.0 - pre_drop_rate
        mean_clos = float(np.mean([r["closure"] for r in obj_res]))
        mean_err = float(np.mean([r["final_err_deg"] for r in obj_res]))
        mean_steps = float(np.mean([r["steps"] for r in obj_res]))
        emi = retention * mean_clos
        
        summary_by_obj[obj_names[obj_id]] = {
            "n": n,
            "succ10": succ10_rate,
            "succ05": succ05_rate,
            "succ15": succ15_rate,
            "retention": retention,
            "closure": mean_clos,
            "mean_err": mean_err,
            "steps": mean_steps,
            "emi": emi,
        }
        
    total_n = len(results)
    tot_succ10 = sum(1 for r in results if r["succ10"]) / total_n
    tot_succ05 = sum(1 for r in results if r["succ05"]) / total_n
    tot_succ15 = sum(1 for r in results if r["succ15"]) / total_n
    tot_pre_drop = sum(1 for r in results if r["dropped_before_success"]) / total_n
    tot_ret = 1.0 - tot_pre_drop
    tot_clos = float(np.mean([r["closure"] for r in results]))
    tot_err = float(np.mean([r["final_err_deg"] for r in results]))
    tot_steps = float(np.mean([r["steps"] for r in results]))
    tot_emi = tot_ret * tot_clos
    
    overall = {
        "n": total_n,
        "succ10": tot_succ10,
        "succ05": tot_succ05,
        "succ15": tot_succ15,
        "retention": tot_ret,
        "closure": tot_clos,
        "mean_err": tot_err,
        "mean_steps": tot_steps,
        "emi": tot_emi,
    }
    
    return summary_by_obj, overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model on 10% Tolerance Benchmark")
    parser.add_argument("--episodes_per_obj", type=int, default=100, help="Episodes per object (default: 100)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()
    
    scene_xml = os.path.join(PROJECT_ROOT, "in_hand_manipulation", "simulation", "scene_cube.xml")
    
    models = {
        "Best Model (Iter 1900 / ppo_best_actor.pth)": os.path.join(PROJECT_ROOT, "stage2_unified", "ppo_best_actor.pth"),
        "Latest Model (Iter 19k+ / ppo_latest_actor.pth)": os.path.join(PROJECT_ROOT, "stage2_unified", "ppo_latest_actor.pth"),
    }
    
    print("\n" + "="*85)
    print(f"BENCHMARK ON 10% RELATIVE TOLERANCE ({args.episodes_per_obj} EPISODES PER OBJECT)")
    print(f"Tolerance Requirement: Error <= 10.0% of Initial Angle (Sub-6° Precision on Nominal 60° Pivots)")
    print(f"Rotation Envelope: 90° Large-Angle (0.15 to 1.57 rad / 8.6° to 90.0°)")
    print("="*85 + "\n")
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: Checkpoint not found at {path}")
            continue
            
        print(f"\n---> Benchmarking: {name}")
        print(f"     Checkpoint Path: {path}")
        t0 = time.time()
        by_obj, overall = evaluate_model(path, scene_xml, args.episodes_per_obj, args.workers)
        elap = time.time() - t0
        
        print("\n" + "-"*85)
        print(f"OVERALL RESULTS FOR {name} ({overall['n']} Test Episodes in {elap:.1f}s):")
        print(f"  * Success Rate @ 10% Tolerance : {overall['succ10']*100:6.2f}%")
        print(f"  * Success Rate @ 15% Tolerance : {overall['succ15']*100:6.2f}%")
        print(f"  * Success Rate @  5% Tolerance : {overall['succ05']*100:6.2f}%")
        print(f"  * Pre-Goal Retention Rate      : {overall['retention']*100:6.2f}%")
        print(f"  * Rotational Closure           : {overall['closure']*100:6.2f}%")
        print(f"  * Effective Manipulation Index : {overall['emi']:.4f}")
        print(f"  * Mean Angular Error           : {overall['mean_err']:6.1f}°")
        print(f"  * Mean Episode Steps           : {overall['mean_steps']:6.1f}")
        print("-"*85)
        
        print("\nPER-OBJECT DIAGNOSTIC BREAKDOWN (100 Episodes Each):")
        print(f"{'Object':<12} | {'Succ (@10%)':<12} | {'Succ (@15%)':<12} | {'Retention':<10} | {'Closure':<10} | {'EMI':<8} | {'Mean Err':<10}")
        print("-" * 85)
        for obj_name, s in by_obj.items():
            print(f"{obj_name:<12} | {s['succ10']*100:10.1f}% | {s['succ15']*100:10.1f}% | {s['retention']*100:8.1f}% | {s['closure']*100:8.1f}% | {s['emi']:.4f} | {s['mean_err']:8.1f}°")
        print("-" * 85 + "\n")


if __name__ == "__main__":
    main()

