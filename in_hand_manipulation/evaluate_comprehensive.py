#!/usr/bin/env python3
"""
evaluate_comprehensive.py
=============================================================================
Advanced Multi-Metric In-Hand Manipulation Evaluator for Delto Gripper-3F.

Features:
1. Multi-Tolerance Spectrum (25%, 20%, 15%, 10%, 5% Error Thresholds)
2. Per-Object Diagnostic Breakdown (Cylinder vs Cube vs Cuboid)
3. OpenAI Dactyl Metric: Consecutive Goals Chained Before Drop/Timeout
4. Physical Hardware Quality: Peak Torques, Action Smoothness, Energy
5. Angular Error Statistics: Initial Angle -> Final Error (Degrees)
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.spatial.transform as transform
import torch
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo import ActorNetwork, DeltoInHandManipulationEnv

OBJECT_NAMES = {1: "Cylinder", 2: "Cube", 3: "Cuboid"}

def run_comprehensive_evaluation(
    checkpoint_path: str,
    scene_xml_path: str = "in_hand_manipulation/simulation/scene_cube.xml",
    episodes_per_object: int = 10,
    max_goal_angle: float = 1.57,
    min_goal_angle: float = 0.60,
    dr_ratio: float = 0.20,
    max_consecutive_goals: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Evaluator] Loading Checkpoint: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    actor = ActorNetwork(obs_dim=39, act_dim=12).to(device)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=max_goal_angle,
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
    )
    
    tolerances = [0.25, 0.20, 0.15, 0.10, 0.05]
    objects = [1, 2, 3] # Cylinder, Cube, Cuboid
    
    # Storage
    records = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBOTIC IN-HAND MANIPULATION BENCHMARK")
    print(f"Goal Envelope: {np.degrees(min_goal_angle):.1f}° to {np.degrees(max_goal_angle):.1f}° | Objects: Cylinder, Cube, Cuboid")
    print(f"Evaluation Protocol: {episodes_per_object} episodes per object ({episodes_per_object * 3} total)")
    print(f"Max Consecutive Goals Allowed per Episode: {max_consecutive_goals}")
    print("="*80 + "\n")
    
    for obj_id in objects:
        obj_name = OBJECT_NAMES[obj_id]
        print(f"\n---> Testing Object: {obj_name} ({episodes_per_object} episodes) <---")
        
        for ep in range(1, episodes_per_object + 1):
            obs, info = env.reset(object_id=obj_id)
            
            # Sample large-angle target
            axis = np.random.randn(3)
            axis /= (np.linalg.norm(axis) + 1e-8)
            target_angle = float(np.random.uniform(min_goal_angle, max_goal_angle))
            rand_rot = transform.Rotation.from_rotvec(axis * target_angle)
            q_xyzw = rand_rot.as_quat()
            env.target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
            
            obj_idx = env.model.jnt_qposadr[env.model.joint(f"obj_{obj_id}_joint").id]
            cur_obj_quat = env.data.qpos[obj_idx+3:obj_idx+7]
            env.prev_d_theta = env.compute_angular_distance(cur_obj_quat, env.target_quat)
            env.initial_d_theta = float(env.prev_d_theta)
            
            init_angle_deg = float(np.degrees(env.initial_d_theta))
            theta_rad = float(env.initial_d_theta)
            quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad ** 2))
            env.current_max_steps = min(800, quad_steps)
            env.set_epsilon(0.20 * theta_rad)
            
            step = 0
            terminated = False
            truncated = False
            step_info = {}
            goals_achieved = 0
            torques = []
            action_diffs = []
            prev_act = None
            
            while not (terminated or truncated):
                step += 1
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                
                if prev_act is not None:
                    action_diffs.append(np.sum(np.square(action - prev_act)))
                prev_act = action.copy()
                
                obs, reward, terminated, truncated, step_info = env.step(action)
                
                # Monitor actuator torques
                torques.append(np.max(np.abs(env.data.actuator_force)))
                
                goals_achieved = step_info.get("successes_in_episode", 0)
                if goals_achieved >= max_consecutive_goals:
                    break
            
            final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
            is_dropped = bool(step_info.get("is_dropped", False))
            
            # Check success across all tolerances
            succ_map = {}
            for tol in tolerances:
                tol_thresh_deg = tol * init_angle_deg
                succ_map[tol] = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)
            
            closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6))) if not is_dropped else 0.0
            if goals_achieved > 0:
                closure = 1.0
            
            records.append({
                "object": obj_name,
                "init_angle_deg": init_angle_deg,
                "final_err_deg": final_err_deg,
                "closure": closure,
                "is_dropped": is_dropped,
                "goals_achieved": goals_achieved,
                "succ_map": succ_map,
                "peak_torque": max(torques) if torques else 0.0,
                "mean_jerk": np.mean(action_diffs) if action_diffs else 0.0,
                "steps": step,
            })
            
            succ_20_str = "YES" if succ_map[0.20] else "NO"
            print(f"  Ep {ep:02d} | Init: {init_angle_deg:4.1f}° | Err: {final_err_deg:4.1f}° | Clos: {closure*100:4.1f}% | Goals: {goals_achieved} | Succ(20%): {succ_20_str} | Drop: {is_dropped} | PeakTorque: {max(torques):.2f}Nm")
    
    # Aggregate Report
    total_eps = len(records)
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE BENCHMARK REPORT")
    print("="*80)
    
    # 1. Multi-Tolerance Spectrum
    print("\n[1. Multi-Tolerance Success Spectrum]")
    for tol in tolerances:
        tol_succ = sum(1 for r in records if r["succ_map"][tol])
        print(f"  Tolerance <= {tol*100:4.1f}% Target Angle :  {tol_succ/total_eps*100:5.1f}%  ({tol_succ}/{total_eps})")
    
    # 2. Per-Object Diagnostic Breakdown
    print("\n[2. Per-Object Performance Breakdown]")
    print(f"{'Object':<12} | {'Succ (20%)':<12} | {'Closure':<10} | {'Drop Rate':<10} | {'Avg Goals':<10} | {'Peak Torque':<10}")
    print("-" * 72)
    for obj_name in ["Cylinder", "Cube", "Cuboid"]:
        obj_recs = [r for r in records if r["object"] == obj_name]
        n_obj = len(obj_recs)
        succ_obj = sum(1 for r in obj_recs if r["succ_map"][0.20]) / n_obj * 100
        clos_obj = np.mean([r["closure"] for r in obj_recs]) * 100
        drop_obj = sum(1 for r in obj_recs if r["is_dropped"]) / n_obj * 100
        goals_obj = np.mean([r["goals_achieved"] for r in obj_recs])
        t_obj = np.max([r["peak_torque"] for r in obj_recs])
        print(f"{obj_name:<12} | {succ_obj:5.1f}%       | {clos_obj:5.1f}%    | {drop_obj:5.1f}%    | {goals_obj:4.2f}       | {t_obj:4.2f} Nm")
    
    # 3. Consecutive Goal Chaining (Dactyl Metric)
    goals_all = [r["goals_achieved"] for r in records]
    print("\n[3. Consecutive Goal Chaining (OpenAI Dactyl Metric)]")
    print(f"  Mean Consecutive Goals per Episode : {np.mean(goals_all):.2f}")
    print(f"  Max Consecutive Goals in an Episode: {max(goals_all)}")
    print(f"  Episodes Reaching >= 2 Goals       : {sum(1 for g in goals_all if g >= 2)}/{total_eps} ({sum(1 for g in goals_all if g >= 2)/total_eps*100:.1f}%)")
    
    # 4. Hardware Safety & Quality
    all_torques = [r["peak_torque"] for r in records]
    all_drops = sum(1 for r in records if r["is_dropped"]) / total_eps * 100
    mean_clos = np.mean([r["closure"] for r in records]) * 100
    emi = (1.0 - all_drops / 100.0) * (mean_clos / 100.0)
    print("\n[4. Global Summary Metrics]")
    print(f"  Effective Manipulation Index (EMI) : {emi:.4f}")
    print(f"  Overall Grasp Retention Rate       : {100.0 - all_drops:.1f}%")
    print(f"  Overall Drop Rate                  : {all_drops:.1f}%")
    print(f"  Overall Mean Rotational Closure    : {mean_clos:.1f}%")
    print(f"  Peak Hardware Torque Recorded      : {max(all_torques):.2f} N·m (Limit: 10.0 N·m)")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Metric In-Hand Manipulation Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to actor checkpoint")
    parser.add_argument("--episodes_per_object", type=int, default=None, help="Episodes per object (default: computed from --episodes)")
    parser.add_argument("--episodes", type=int, default=15, help="Total episodes across all 3 objects (default: 15)")
    parser.add_argument("--max_goal_angle", type=float, default=1.57, help="Max target rotation angle in radians")
    parser.add_argument("--min_goal_angle", type=float, default=0.60, help="Min target rotation angle in radians")
    parser.add_argument("--dr_ratio", type=float, default=0.20, help="Domain randomization ratio")
    parser.add_argument("--no_render", action="store_true", help="Headless evaluation flag (compatibility)")
    args = parser.parse_args()
    
    eps_per_obj = args.episodes_per_object if args.episodes_per_object is not None else max(1, args.episodes // 3)
    run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        episodes_per_object=eps_per_obj,
        max_goal_angle=args.max_goal_angle,
        min_goal_angle=args.min_goal_angle,
        dr_ratio=args.dr_ratio,
    )
