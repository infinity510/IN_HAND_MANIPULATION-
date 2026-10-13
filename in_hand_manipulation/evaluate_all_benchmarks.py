#!/usr/bin/env python3
"""
evaluate_all_benchmarks.py
=============================================================================
Comprehensive Benchmarking Suite for In-Hand Manipulation.
Evaluates all golden models under the current simulation physics settings
(palm_hearth collision support, calibrated 120°/100° joint limits, flat-shelf penalty).
Runs 100 evaluation episodes PER OBJECT (100 Cylinder, 100 Cube, 100 Cuboid)
for a total of 300 test episodes per model under the 90° rotation envelope.
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
_worker_min_goal_angle = 0.60
_worker_epsilon_ratio = 0.15


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
    is_dropped = bool(step_info.get("is_dropped", False))
    goals_achieved = step_info.get("successes_in_episode", 0)
    
    # Tolerances
    tol15_deg = 0.15 * init_angle_deg
    tol20_deg = 0.20 * init_angle_deg
    tol25_deg = 0.25 * init_angle_deg
    
    succ15 = bool(goals_achieved > 0 or final_err_deg <= tol15_deg)
    succ20 = bool(goals_achieved > 0 or final_err_deg <= tol20_deg)
    succ25 = bool(goals_achieved > 0 or final_err_deg <= tol25_deg)
    
    if goals_achieved > 0:
        closure = 1.0
    elif is_dropped:
        closure = 0.0
    else:
        closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))
        
    return {
        "object_id": obj_id,
        "succ15": succ15,
        "succ20": succ20,
        "succ25": succ25,
        "dropped": is_dropped,
        "closure": closure,
        "final_err_deg": final_err_deg,
        "init_angle_deg": init_angle_deg,
        "steps": step,
    }


def evaluate_model_on_all_objects(ckpt_path: str, scene_xml_path: str, episodes_per_obj: int = 100, num_workers: int = 6):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    init_args = (state_dict, obs_mean, obs_std, scene_xml_path, 1.57, 0.60, 0.15)
    
    tasks = []
    base_seed = 42000
    for obj_id in [1, 2, 3]:  # 1=Cylinder, 2=Cube, 3=Cuboid
        for ep in range(episodes_per_obj):
            tasks.append((base_seed + obj_id * 1000 + ep, obj_id))
            
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_bench_worker, initargs=init_args) as pool:
        results = pool.map(_eval_single_run, tasks)
        
    obj_names = {1: "Cylinder", 2: "Cube", 3: "Cuboid"}
    summary_by_obj = {}
    
    for obj_id in [1, 2, 3]:
        obj_res = [r for r in results if r["object_id"] == obj_id]
        n = len(obj_res)
        succ15_rate = sum(1 for r in obj_res if r["succ15"]) / n
        succ20_rate = sum(1 for r in obj_res if r["succ20"]) / n
        succ25_rate = sum(1 for r in obj_res if r["succ25"]) / n
        drop_rate = sum(1 for r in obj_res if r["dropped"]) / n
        retention = 1.0 - drop_rate
        closure = float(np.mean([r["closure"] for r in obj_res]))
        mean_err = float(np.mean([r["final_err_deg"] for r in obj_res]))
        mean_steps = float(np.mean([r["steps"] for r in obj_res]))
        emi15 = retention * closure
        
        summary_by_obj[obj_names[obj_id]] = {
            "n": n,
            "succ15": succ15_rate,
            "succ20": succ20_rate,
            "succ25": succ25_rate,
            "drop": drop_rate,
            "retention": retention,
            "closure": closure,
            "mean_err": mean_err,
            "mean_steps": mean_steps,
            "emi": emi15,
        }
        
    total_n = len(results)
    tot_succ15 = sum(1 for r in results if r["succ15"]) / total_n
    tot_succ20 = sum(1 for r in results if r["succ20"]) / total_n
    tot_succ25 = sum(1 for r in results if r["succ25"]) / total_n
    tot_drop = sum(1 for r in results if r["dropped"]) / total_n
    tot_ret = 1.0 - tot_drop
    tot_clos = float(np.mean([r["closure"] for r in results]))
    tot_err = float(np.mean([r["final_err_deg"] for r in results]))
    tot_steps = float(np.mean([r["steps"] for r in results]))
    tot_emi = tot_ret * tot_clos
    
    overall = {
        "n": total_n,
        "succ15": tot_succ15,
        "succ20": tot_succ20,
        "succ25": tot_succ25,
        "drop": tot_drop,
        "retention": tot_ret,
        "closure": tot_clos,
        "mean_err": tot_err,
        "mean_steps": tot_steps,
        "emi": tot_emi,
    }
    
    return summary_by_obj, overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate All Models on 100 Episodes per Object")
    parser.add_argument("--episodes_per_obj", type=int, default=100, help="Episodes per object (default: 100)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()
    
    scene_xml = os.path.join(PROJECT_ROOT, "in_hand_manipulation", "simulation", "scene_cube.xml")
    
    models = {
        "Stage 1 Golden (Precision Baseline)": os.path.join(PROJECT_ROOT, "stage1", "stage1_best_model.pth"),
        "Stage 2A Golden (25% Relative Tolerance)": os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps25", "stage2a_golden.pth"),
        "Stage 2B Golden (20% Relative Tolerance)": os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps20", "stage2b_golden.pth"),
        "Stage 2C Active Best (15% Tolerance Pipeline)": os.path.join(PROJECT_ROOT, "checkpoints_stage2_eps15", "ppo_best_actor.pth"),
    }
    
    print("\n" + "="*85)
    print(f"BENCHMARKING ALL MODELS UNDER CURRENT SETTINGS ({args.episodes_per_obj} EPISODES PER OBJECT)")
    print(f"Current Physics: palm_hearth cylinder, 120° m3, 100° m4, flat-shelf penalty (thresh 3.20 rad)")
    print(f"Envelope: 90° Large-Angle (0.60 to 1.57 rad), Current Stage 2C Tolerance = 15%")
    print("="*85 + "\n")
    
    all_results = {}
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: Checkpoint not found at {path}")
            continue
            
        print(f"\n---> Evaluating: {name}")
        print(f"     Path: {path}")
        t0 = time.time()
        by_obj, overall = evaluate_model_on_all_objects(path, scene_xml, args.episodes_per_obj, args.workers)
        elap = time.time() - t0
        all_results[name] = {"by_obj": by_obj, "overall": overall, "time": elap}
        
        print(f"     [Overall ({overall['n']} episodes in {elap:.1f}s)]: "
              f"Succ@15%: {overall['succ15']*100:.1f}% | Succ@20%: {overall['succ20']*100:.1f}% | "
              f"Retention: {overall['retention']*100:.1f}% | Closure: {overall['closure']*100:.1f}% | "
              f"EMI: {overall['emi']:.3f} | Error: {overall['mean_err']:.1f}°")
              
        for obj_name, stats in by_obj.items():
            print(f"       • {obj_name:8s} ({stats['n']} eps): "
                  f"Succ@15%: {stats['succ15']*100:5.1f}% | Succ@20%: {stats['succ20']*100:5.1f}% | "
                  f"Ret: {stats['retention']*100:5.1f}% | Clos: {stats['closure']*100:5.1f}% | "
                  f"EMI: {stats['emi']:.3f} | Err: {stats['mean_err']:4.1f}°")

    print("\n\n" + "="*85)
    print("FINAL COMPARISON MATRIX (100 EPISODES PER OBJECT, 300 EPISODES TOTAL)")
    print("="*85)
    print(f"{'Model':<38} | {'Succ@15%':<9} | {'Succ@20%':<9} | {'Retention':<9} | {'Closure':<9} | {'EMI':<6} | {'Err (deg)':<9}")
    print("-" * 105)
    for name, res in all_results.items():
        o = res["overall"]
        print(f"{name:<38} | {o['succ15']*100:6.1f}%   | {o['succ20']*100:6.1f}%   | {o['retention']*100:6.1f}%   | {o['closure']*100:6.1f}%   | {o['emi']:<6.3f} | {o['mean_err']:<6.1f}°")
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
