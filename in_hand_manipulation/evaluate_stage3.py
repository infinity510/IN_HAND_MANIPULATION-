#!/usr/bin/env python3
"""
evaluate_stage3.py
=============================================================================
STAGE 3 EVALUATION: Full SO(3) Extreme Flips & Continuous In-Hand Manipulation
-----------------------------------------------------------------------------
Evaluates the in-hand manipulation policy on arbitrary 3D target orientations
sampled uniformly across full SO(3) (up to 180° complete object inversion).
Tests multi-phase finger gaiting cycles and continuous multi-goal chaining
(OpenAI Dactyl benchmark).

Key Characteristics:
  - Max Goal Angle: Full SO(3) (up to pi rad = 180.0° complete inversion)
  - Target Sampling: Uniform Haar measure over SO(3) via random unit quaternions
  - Continuous Resampling: Immediately resamples new target when goal reached
  - Horizon: Full extended horizon [300 + 200*theta + 150*theta^2] up to 800 steps (16.0s)
  - Metric: Effective Manipulation Index (EMI) + Multi-Goal Achievement Count
  - Visual: Real-time 3D Green Target Arrow + 1.0s settling pause between episodes
=============================================================================
"""

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import scipy.spatial.transform as transform
import torch
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo import ActorNetwork, DeltoInHandManipulationEnv


def draw_target_arrow(viewer, target_quat: np.ndarray, obj_pos: np.ndarray):
    """Renders a dynamic green 3D orientation arrow in the MuJoCo viewer pointing along target frame."""
    if viewer is None or not hasattr(viewer, "user_scn") or viewer.user_scn is None:
        return
    viewer.user_scn.ngeom = 0
    r = transform.Rotation.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
    mat = r.as_matrix().flatten().astype(np.float64)
    arrow_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.08], dtype=np.float64)
    arrow_size = np.array([0.005, 0.005, 0.07], dtype=np.float64)
    arrow_rgba = np.array([0.1, 0.95, 0.2, 0.9], dtype=np.float32)

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, arrow_size, arrow_pos, mat, arrow_rgba)
    viewer.user_scn.ngeom = 1


def evaluate_stage3(
    checkpoint_path: str,
    scene_xml_path: str,
    render: bool = True,
    num_episodes: int = 15,
    object_id: Optional[int] = None,
    dr_ratio: float = 0.20,
    inter_episode_delay: float = 1.0,
    success_ratio: float = 0.10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 3 Evaluator] Loading model: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    actor = ActorNetwork(obs_dim=39, act_dim=12).to(device)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    # max_goal_angle = np.pi activates full SO(3) uniform sampling
    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=300,
        max_goal_angle=float(np.pi),
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=200,
        max_steps_cap=800,
    )
    
    print("\n" + "="*70)
    print("STAGE 3 EVALUATION: FULL SO(3) EXTREME FLIPS (theta up to 180.0°)")
    print(f"Goal Distribution: Uniform Haar measure over SO(3) | Object: {object_id or 'Random (1-3)'}")
    print(f"Success Threshold Law: Proportional {success_ratio * 100:.0f}% of required rotation angle")
    print(f"Chaining Feature: Continuous goal resampling enabled (OpenAI Dactyl Multi-Goal chaining)")
    print(f"Horizon Scaling: [300 + 200*theta + 150*theta^2 + 200/goal] capped at 800 steps (16.0s)")
    print(f"Settling Delay: {inter_episode_delay:.1f}s between episodes with live green target arrow")
    print("="*70 + "\n")
    
    successes = 0
    drops = 0
    total_goals_chained = 0
    errors_deg = []
    closures = []
    episode_lengths = []
    
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    try:
        for ep in range(1, num_episodes + 1):
            if viewer is not None and not viewer.is_running():
                print("\n[Notice] MuJoCo viewer closed by user. Ending evaluation early.")
                break

            obs, info = env.reset(object_id=object_id)
            init_angle_deg = info.get("initial_angle_deg", np.degrees(info.get("d_theta", 0.0)))
            init_angle_rad = np.radians(init_angle_deg)
            
            eval_epsilon_rad = max(0.015, success_ratio * init_angle_rad)
            eval_epsilon_deg = float(np.degrees(eval_epsilon_rad))
            env.set_epsilon(eval_epsilon_rad)
            
            ep_reward = 0.0
            step = 0
            terminated = False
            truncated = False
            step_info = {}
            goals_in_ep = 0
            
            max_steps = env.current_max_steps
            print(f"\n[Episode {ep:02d}/{num_episodes:02d}] Object: {env.active_object_id} | "
                  f"Initial Target Angle: {init_angle_deg:5.1f}° | "
                  f"Success Threshold ({success_ratio * 100:.0f}%): <= {eval_epsilon_deg:4.2f}° | "
                  f"Horizon: {max_steps} steps ({max_steps * 0.02:.2f}s)")
            
            obj_idx = env.model.jnt_qposadr[env.model.joint(f"obj_{env.active_object_id}_joint").id]
            
            while not (terminated or truncated):
                step += 1
                step_start = time.time()
                
                if viewer is not None and not viewer.is_running():
                    return
                    
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                    
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                
                # Check if a new goal was scored during this step
                if step_info.get("successes_in_episode", 0) > goals_in_ep:
                    goals_in_ep = step_info["successes_in_episode"]
                    print(f"    >>> [GOAL {goals_in_ep} ACHIEVED!] Continuously resampled new SO(3) target goal!")
                
                if viewer is not None:
                    cur_obj_pos = env.data.qpos[obj_idx:obj_idx+3]
                    draw_target_arrow(viewer, env.target_quat, cur_obj_pos)
                    viewer.sync()
                    
                    elapsed = time.time() - step_start
                    if 0.02 - elapsed > 0:
                        time.sleep(0.02 - elapsed)
                        
            final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
            closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))
            is_succ = bool(final_err_deg <= eval_epsilon_deg or step_info.get("is_success", False) or goals_in_ep > 0)
            is_drop = bool(step_info.get("is_dropped", False))
            
            errors_deg.append(final_err_deg)
            closures.append(closure)
            episode_lengths.append(step)
            total_goals_chained += goals_in_ep
            if is_succ:
                successes += 1
            if is_drop:
                drops += 1
                
            status_str = f"SUCCESS (Goals Chained: {goals_in_ep})" if is_succ else ("DROPPED" if is_drop else f"TIMEOUT ({final_err_deg:.1f}° > {eval_epsilon_deg:.1f}°)")
            print(f"  Finished: Steps: {step:3d}/{max_steps} | "
                  f"Final Error: {final_err_deg:5.2f}° | "
                  f"Rotational Closure: {closure * 100:5.1f}% | "
                  f"Status: {status_str}")
            
            if ep < num_episodes:
                if viewer is not None and viewer.is_running():
                    delay_start = time.time()
                    cur_obj_pos = env.data.qpos[obj_idx:obj_idx+3]
                    while time.time() - delay_start < inter_episode_delay and viewer.is_running():
                        draw_target_arrow(viewer, env.target_quat, cur_obj_pos)
                        viewer.sync()
                        time.sleep(0.02)
                else:
                    time.sleep(inter_episode_delay)

    finally:
        if viewer is not None:
            viewer.close()

    if episode_lengths:
        n = len(episode_lengths)
        succ_rate = successes / n
        drop_rate = drops / n
        retention_rate = max(0.0, 1.0 - drop_rate)
        mean_closure = float(np.mean(closures))
        emi = retention_rate * mean_closure
        avg_chained = total_goals_chained / n
        
        print("\n" + "="*70)
        print(f"STAGE 3 BENCHMARK SUMMARY ({n} Episodes Completed):")
        print(f"Effective Manipulation Index (EMI): {emi:.4f}  (Retention: {retention_rate*100:.1f}% x Closure: {mean_closure*100:.1f}%)")
        print(f"Success Rate (Goals Reached):       {succ_rate * 100:5.1f}% ({successes}/{n})")
        print(f"Average Goals Chained per Episode:  {avg_chained:.2f} goals/ep (Total: {total_goals_chained})")
        print(f"Mean Rotational Closure:            {mean_closure * 100:5.1f}%")
        print(f"Drop Rate (Grasp Stability):        {drop_rate * 100:5.1f}% ({drops}/{n})")
        print(f"Mean Episode Steps:                 {np.mean(episode_lengths):.1f} steps ({np.mean(episode_lengths)*0.02:.2f}s)")
        print(f"Mean Angular Error:                 {np.mean(errors_deg):.2f}° +/- {np.std(errors_deg):.2f}°")
        print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Evaluation: Full SO(3) Extreme Flips & Continuous Chaining")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ppo/ppo_best_actor.pth", help="Path to model checkpoint")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml", help="Path to scene XML")
    parser.add_argument("--episodes", type=int, default=15, help="Number of evaluation episodes (default: 15)")
    parser.add_argument("--object_id", type=int, default=0, help="Target object: 0=random, 1=cylinder, 2=cube, 3=cuboid (default: 0)")
    parser.add_argument("--dr_ratio", type=float, default=0.20, help="Domain randomization ratio (default: 0.20)")
    parser.add_argument("--delay", type=float, default=1.0, help="Inter-episode delay in seconds (default: 1.0s)")
    parser.add_argument("--success_ratio", type=float, default=0.10, help="Success threshold ratio relative to target angle (default: 0.10 = 10%%)")
    parser.add_argument("--no_render", action="store_true", help="Disable 3D viewer for fast headless benchmarking")
    args = parser.parse_args()
    
    evaluate_stage3(
        checkpoint_path=args.checkpoint,
        scene_xml_path=args.scene_xml,
        render=not args.no_render,
        num_episodes=args.episodes,
        object_id=args.object_id if args.object_id > 0 else None,
        dr_ratio=args.dr_ratio,
        inter_episode_delay=args.delay,
        success_ratio=args.success_ratio,
    )
