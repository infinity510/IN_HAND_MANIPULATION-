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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo import ActorNetwork, DeltoInHandManipulationEnv


def evaluate(checkpoint_path: str, scene_xml_path: str, render: bool = True, num_episodes: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]
    
    # Initialize Actor
    actor = ActorNetwork(obs_dim=39, act_dim=12).to(device)
    # Check if keys are nested or raw
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250
    )
    
    print("\n" + "="*60)
    print("STARTING PPO POLICY EVALUATION")
    print("="*60 + "\n")
    
    successes = 0
    errors_deg = []
    
    if render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            for ep in range(1, num_episodes + 1):
                obs, info = env.reset()
                ep_reward = 0.0
                print(f"\n[Episode {ep}/{num_episodes}] Object: {info['object_id']} | Target Quat: {np.round(info['target_quat'], 2)}")
                
                for step in range(250):
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
                        
                    if terminated or truncated:
                        final_err_deg = step_info["d_theta"] * (180.0 / np.pi)
                        errors_deg.append(final_err_deg)
                        if step_info["is_success"]:
                            successes += 1
                        print(f"  Finished: Step {step+1} | Reward: {ep_reward:.1f} | Error: {final_err_deg:.2f}° | Success: {step_info['is_success']} | Dropped: {step_info['is_dropped']}")
                        break
    else:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            ep_reward = 0.0
            for step in range(250):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    final_err_deg = step_info["d_theta"] * (180.0 / np.pi)
                    errors_deg.append(final_err_deg)
                    if step_info["is_success"]:
                        successes += 1
                    print(f"[Episode {ep:02d}] Error: {final_err_deg:5.2f}° | Success: {step_info['is_success']} | Dropped: {step_info['is_dropped']}")
                    break
                    
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY ({num_episodes} Episodes):")
    print(f"Success Rate: {successes / num_episodes * 100:.1f}%")
    print(f"Mean Angular Error: {np.mean(errors_deg):.2f}° +/- {np.std(errors_deg):.2f}°")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ppo/ppo_best_actor.pth")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        scene_xml_path=args.scene_xml,
        render=not args.no_render,
        num_episodes=args.episodes
    )

