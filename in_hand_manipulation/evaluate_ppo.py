#!/usr/bin/env python3
"""
evaluate_ppo.py
=============================================================================
Live Evaluation & Benchmark for PPO Fine-Tuned Delto Hand Policy.
Visualizes in-hand manipulation in MuJoCo viewer and measures orientation error.
=============================================================================
"""

import os
import time
import argparse
import numpy as np
import scipy.spatial.transform as transform
import torch
import mujoco
import mujoco.viewer

import sys
from typing import Optional
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo import ActorNetwork, DeltoInHandManipulationEnv


def evaluate(
    checkpoint_path: str,
    scene_xml_path: str,
    render: bool = True,
    num_episodes: int = 10,
    max_goal_angle: float = 0.80,
    object_id: Optional[int] = None,
    dr_ratio: float = 0.20
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
    
    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=max_goal_angle,
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800
    )
    
    print("\n" + "="*60)
    print("STARTING PPO POLICY EVALUATION")
    print(f"Max Goal Angle: {max_goal_angle:.2f} rad ({np.degrees(max_goal_angle):.1f}°) | Target Object: {object_id or 'Random (1-3)'}")
    print("="*60 + "\n")
    
    successes = 0
    drops = 0
    errors_deg = []
    episode_lengths = []
    
    if render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            for ep in range(1, num_episodes + 1):
                obs, info = env.reset(object_id=object_id)
                ep_reward = 0.0
                step = 0
                terminated = False
                truncated = False
                step_info = {}
                print(f"\n[Episode {ep}/{num_episodes}] Object: {info['object_id']} | Target Quat: {np.round(info['target_quat'], 2)}")
                
                while not (terminated or truncated):
                    step += 1
                    step_start = time.time()
                    if not viewer.is_running():
                        return
                        
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = actor(obs_t).squeeze(0).cpu().numpy()
                        
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    ep_reward += reward
                    viewer.sync()
                    
                    # Target 50 Hz loop
                    elapsed = time.time() - step_start
                    if 0.02 - elapsed > 0:
                        time.sleep(0.02 - elapsed)
                        
                final_err_deg = step_info["d_theta"] * (180.0 / np.pi)
                errors_deg.append(final_err_deg)
                episode_lengths.append(step)
                if step_info.get("is_success", False):
                    successes += 1
                if step_info.get("is_dropped", False):
                    drops += 1
                print(f"  Finished: Step {step} | Reward: {ep_reward:.1f} | Error: {final_err_deg:.2f}° | Success: {step_info.get('is_success', False)} | Dropped: {step_info.get('is_dropped', False)}")
    else:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset(object_id=object_id)
            ep_reward = 0.0
            step = 0
            terminated = False
            truncated = False
            step_info = {}
            while not (terminated or truncated):
                step += 1
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                
            final_err_deg = step_info["d_theta"] * (180.0 / np.pi)
            errors_deg.append(final_err_deg)
            episode_lengths.append(step)
            if step_info.get("is_success", False):
                successes += 1
            if step_info.get("is_dropped", False):
                drops += 1
            print(f"[Episode {ep:02d}] Steps: {step:3d} | Error: {final_err_deg:5.2f}° | Success: {step_info.get('is_success', False)} | Dropped: {step_info.get('is_dropped', False)}")
                    
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY ({num_episodes} Episodes):")
    print(f"Success Rate:      {successes / num_episodes * 100:5.1f}% ({successes}/{num_episodes})")
    print(f"Drop Rate:         {drops / num_episodes * 100:5.1f}% ({drops}/{num_episodes})")
    print(f"Mean Steps:        {np.mean(episode_lengths):.1f} steps ({np.mean(episode_lengths)*0.02:.2f}s)")
    print(f"Mean Angular Error: {np.mean(errors_deg):.2f}° +/- {np.std(errors_deg):.2f}°")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ppo/ppo_best_actor.pth")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_goal_angle", type=float, default=0.80, help="Max target rotation angle in radians (default: 0.80 rad ~= 45 deg)")
    parser.add_argument("--object_id", type=int, default=0, help="Target object: 0=random, 1=cylinder, 2=cube, 3=cuboid (default: 0)")
    parser.add_argument("--dr_ratio", type=float, default=0.20, help="Domain randomization ratio (default: 0.20)")
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        scene_xml_path=args.scene_xml,
        render=not args.no_render,
        num_episodes=args.episodes,
        max_goal_angle=args.max_goal_angle,
        object_id=args.object_id if args.object_id > 0 else None,
        dr_ratio=args.dr_ratio
    )

