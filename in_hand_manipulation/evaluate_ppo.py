#!/usr/bin/env python3
"""
evaluate_ppo.py
=============================================================================
Live Evaluation & Benchmark for PPO Fine-Tuned Delto Hand Policy.
Features:
  - Dynamic 3D Green Arrow pointing in the direction of the target orientation.
  - Quadratic angle-dependent episode horizon (scaling with required rotation).
  - 1-second observation delay between successive trajectories.
  - Streamlined, unified execution loop for both visual and headless modes.
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
    """
    Renders a dynamic green 3D orientation arrow in the MuJoCo viewer pointing
    in the direction of the target orientation frame.
    """
    if viewer is None or not hasattr(viewer, "user_scn") or viewer.user_scn is None:
        return
    viewer.user_scn.ngeom = 0
    # MuJoCo quat convention: [w, x, y, z] -> Scipy: [x, y, z, w]
    r = transform.Rotation.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
    mat = r.as_matrix().flatten().astype(np.float64)
    # Position arrow slightly above object (0.08m) so it hovers clearly above contacts
    arrow_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.08], dtype=np.float64)
    arrow_size = np.array([0.005, 0.005, 0.07], dtype=np.float64)  # [radius, radius, length]
    arrow_rgba = np.array([0.1, 0.95, 0.2, 0.9], dtype=np.float32)  # Bright Green

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, arrow_size, arrow_pos, mat, arrow_rgba)
    viewer.user_scn.ngeom = 1


def evaluate(
    checkpoint_path: str,
    scene_xml_path: str,
    render: bool = True,
    num_episodes: int = 10,
    max_goal_angle: float = 0.80,
    object_id: Optional[int] = None,
    dr_ratio: float = 0.20,
    inter_episode_delay: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    # Initialize Actor
    actor = ActorNetwork(obs_dim=39, act_dim=12).to(device)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    # Environment with quadratic angle-dependent horizon
    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=200,
        max_goal_angle=max_goal_angle,
        dr_ratio=dr_ratio,
        steps_per_radian=150,
        bonus_steps_per_success=150,
        max_steps_cap=800,
    )
    
    print("\n" + "="*65)
    print("STARTING PPO POLICY EVALUATION")
    print(f"Max Goal Angle: {max_goal_angle:.2f} rad ({np.degrees(max_goal_angle):.1f}°) | Target Object: {object_id or 'Random (1-3)'}")
    print(f"Angle-Horizon Law: Quadratic Scaling [200 + 150*theta + 150*theta^2] capped at 800 steps")
    print(f"Trajectory Pause: {inter_episode_delay:.1f}s between episodes with live green target arrow")
    print("="*65 + "\n")
    
    successes = 0
    drops = 0
    errors_deg = []
    episode_lengths = []
    
    # Launch viewer context if rendering enabled
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
            ep_reward = 0.0
            step = 0
            terminated = False
            truncated = False
            step_info = {}
            
            init_angle = info.get("initial_angle_deg", np.degrees(info.get("d_theta", 0.0)))
            max_steps = info.get("max_steps", env.current_max_steps)
            print(f"\n[Episode {ep:02d}/{num_episodes:02d}] Object: {info['object_id']} | "
                  f"Initial Target Angle: {init_angle:5.1f}° | "
                  f"Allocated Horizon: {max_steps} steps ({max_steps * 0.02:.2f}s)")
            
            # Helper to get current object world position
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
                
                if viewer is not None:
                    cur_obj_pos = env.data.qpos[obj_idx:obj_idx+3]
                    draw_target_arrow(viewer, env.target_quat, cur_obj_pos)
                    viewer.sync()
                    
                    # Pace to real-time 50 Hz physics
                    elapsed = time.time() - step_start
                    if 0.02 - elapsed > 0:
                        time.sleep(0.02 - elapsed)
                        
            final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
            is_succ = step_info.get("is_success", False)
            is_drop = step_info.get("is_dropped", False)
            
            errors_deg.append(final_err_deg)
            episode_lengths.append(step)
            if is_succ:
                successes += 1
            if is_drop:
                drops += 1
                
            status_str = "SUCCESS" if is_succ else ("DROPPED" if is_drop else "TIME_LIMIT")
            print(f"  Finished: Steps: {step:3d}/{max_steps} | "
                  f"Reward: {ep_reward:7.1f} | "
                  f"Error: {final_err_deg:5.2f}° | "
                  f"Status: {status_str}")
            
            # Inter-episode delay (1.0 second) allowing visual review of final state
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
        print("\n" + "="*65)
        print(f"EVALUATION SUMMARY ({len(episode_lengths)} Episodes Completed):")
        print(f"Success Rate:       {successes / len(episode_lengths) * 100:5.1f}% ({successes}/{len(episode_lengths)})")
        print(f"Drop Rate:          {drops / len(episode_lengths) * 100:5.1f}% ({drops}/{len(episode_lengths)})")
        print(f"Mean Steps:         {np.mean(episode_lengths):.1f} steps ({np.mean(episode_lengths)*0.02:.2f}s)")
        print(f"Mean Angular Error: {np.mean(errors_deg):.2f}° +/- {np.std(errors_deg):.2f}°")
        print("="*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Delto PPO Policy with Visual Target Arrow & Quadratic Horizon")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ppo/ppo_best_actor.pth", help="Path to model checkpoint")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml", help="Path to scene XML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_goal_angle", type=float, default=0.80, help="Max target rotation angle in radians (default: 0.80 rad ~= 45.8 deg)")
    parser.add_argument("--object_id", type=int, default=0, help="Target object: 0=random, 1=cylinder, 2=cube, 3=cuboid (default: 0)")
    parser.add_argument("--dr_ratio", type=float, default=0.20, help="Domain randomization ratio (default: 0.20)")
    parser.add_argument("--delay", type=float, default=1.0, help="Inter-episode delay in seconds (default: 1.0s)")
    parser.add_argument("--no_render", action="store_true", help="Disable 3D viewer for fast headless benchmarking")
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        scene_xml_path=args.scene_xml,
        render=not args.no_render,
        num_episodes=args.episodes,
        max_goal_angle=args.max_goal_angle,
        object_id=args.object_id if args.object_id > 0 else None,
        dr_ratio=args.dr_ratio,
        inter_episode_delay=args.delay,
    )
