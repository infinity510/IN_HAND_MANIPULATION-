#!/usr/bin/env python3
"""
evaluate_bc.py
=============================================================================
Live Evaluation & Benchmark for Behavioral Cloning (BC) Pre-Trained Policy.
Features:
  - Dynamic 3D Green Arrow pointing in the direction of the target orientation.
  - Quadratic angle-dependent episode horizon (scaling with required rotation).
  - 1-second observation delay between successive trajectories.
  - Streamlined, unified execution loop for both visual and headless modes.
  - Full removal of deprecated dependencies (pynput) and redundant code.
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

from train_bc import ActorNetwork


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


def compute_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Computes geodesic SO(3) angular distance between two quaternions in radians."""
    dot = np.abs(np.dot(q1, q2))
    dot_clamped = np.clip(dot, -1.0, 1.0)
    return float(2.0 * np.arccos(dot_clamped))


class BCEvaluator:
    def __init__(self, checkpoint_path: str, scene_xml_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BC checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.obs_mean = payload["obs_mean"].astype(np.float32)
        self.obs_std = payload["obs_std"].astype(np.float32)
        hp = payload.get("hyperparameters", {"obs_dim": 39, "act_dim": 12, "hidden_dims": [256]})
        
        self.actor = ActorNetwork(
            obs_dim=hp["obs_dim"],
            act_dim=hp["act_dim"],
            hidden_dim=hp["hidden_dims"][0]
        ).to(self.device)
        self.actor.load_state_dict(payload["model_state_dict"])
        self.actor.eval()
        
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i).startswith("gripper_")
            and "joint" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name.replace('gripper_', '').replace('_joint', '')}")
            for name in self.joint_names
        ]

    def get_physical_params(self, obj_id: int) -> np.ndarray:
        if obj_id == 1:
            return np.array([0.045, 0.07, 1.0, 3.0], dtype=np.float32)
        elif obj_id == 2:
            return np.array([0.045, 0.045, 1.0, 3.0], dtype=np.float32)
        elif obj_id == 3:
            return np.array([0.055, 0.07, 1.0, 3.0], dtype=np.float32)
        return np.array([0.045, 0.07, 1.0, 3.0], dtype=np.float32)

    def reset_gripper_and_object(self, obj_id: int, max_goal_angle: float):
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset fingers to canonical pinch tripod pose
        initial_pose = [
            ("gripper_f1m1_joint", 0.0), ("gripper_f1m2_joint", 0.0),
            ("gripper_f1m3_joint", 0.8), ("gripper_f1m4_joint", 0.8),
            ("gripper_f2m1_joint", -1.047), ("gripper_f2m2_joint", 0.0),
            ("gripper_f2m3_joint", 0.8), ("gripper_f2m4_joint", 0.8),
            ("gripper_f3m1_joint", 1.047), ("gripper_f3m2_joint", 0.0),
            ("gripper_f3m3_joint", 0.8), ("gripper_f3m4_joint", 0.8),
        ]
        for name, val in initial_pose:
            try:
                j_id = self.model.joint(name).id
                qpos_idx = self.model.jnt_qposadr[j_id]
                vel_idx = self.model.jnt_dofadr[j_id]
                self.data.qpos[qpos_idx] = val
                self.data.qvel[vel_idx] = 0.0
                act_name = f"act_{name.replace('gripper_', '').replace('_joint', '')}"
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if act_id != -1:
                    self.data.ctrl[act_id] = val
            except Exception:
                pass
                
        # Position active object at palm center (0, 0, 0.10) and move others far away
        for i in range(1, 4):
            try:
                idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
                vel_idx = self.model.jnt_dofadr[self.model.joint(f"obj_{i}_joint").id]
                if i == obj_id:
                    self.data.qpos[idx:idx+7] = [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]
                else:
                    self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]
                self.data.qvel[vel_idx:vel_idx+6] = 0.0
            except Exception:
                pass

        # Sample target quaternion
        if max_goal_angle >= (np.pi - 1e-4):
            q_scipy = transform.Rotation.random().as_quat()
            target_quat = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float32)
        else:
            axis = np.random.randn(3)
            axis /= (np.linalg.norm(axis) + 1e-8)
            angle = np.random.uniform(0.30, max_goal_angle)
            rot = transform.Rotation.from_rotvec(axis * angle)
            q_scipy = rot.as_quat()
            target_quat = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)

        return target_quat

    def run_evaluation(
        self,
        num_episodes: int = 10,
        render: bool = True,
        max_goal_angle: float = 0.80,
        object_id: Optional[int] = None,
        inter_episode_delay: float = 1.0,
    ):
        print("\n" + "="*65)
        print("STARTING BEHAVIORAL CLONING (BC) POLICY EVALUATION")
        print(f"Max Goal Angle: {max_goal_angle:.2f} rad ({np.degrees(max_goal_angle):.1f}°) | Target Object: {object_id or 'Random (1-3)'}")
        print(f"Angle-Horizon Law: Quadratic Scaling [200 + 150*theta + 150*theta^2] capped at 800 steps")
        print(f"Trajectory Pause: {inter_episode_delay:.1f}s between episodes with live green target arrow")
        print("="*65 + "\n")
        
        successes = 0
        drops = 0
        errors_deg = []
        episode_lengths = []
        
        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        try:
            for ep in range(1, num_episodes + 1):
                if viewer is not None and not viewer.is_running():
                    print("\n[Notice] MuJoCo viewer closed by user. Ending evaluation early.")
                    break

                curr_obj_id = object_id if object_id in [1, 2, 3] else np.random.choice([1, 2, 3])
                target_quat = self.reset_gripper_and_object(curr_obj_id, max_goal_angle)
                
                obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{curr_obj_id}_joint").id]
                cur_obj_pos = self.data.qpos[obj_idx:obj_idx+3].copy()
                cur_obj_quat = self.data.qpos[obj_idx+3:obj_idx+7].copy()
                
                initial_err = compute_angular_distance(cur_obj_quat, target_quat)
                initial_deg = np.degrees(initial_err)
                
                # Quadratic Horizon Law
                quad_steps = int(200 + 150 * initial_err + 150 * (initial_err ** 2))
                max_steps = min(800, quad_steps)
                
                print(f"\n[Episode {ep:02d}/{num_episodes:02d}] Object: {curr_obj_id} | "
                      f"Initial Target Angle: {initial_deg:5.1f}° | "
                      f"Allocated Horizon: {max_steps} steps ({max_steps * 0.02:.2f}s)")
                
                step = 0
                is_dropped = False
                is_success = False
                final_err_deg = initial_deg
                
                while step < max_steps:
                    step += 1
                    step_start = time.time()
                    
                    if viewer is not None and not viewer.is_running():
                        return

                    # Build 39-dim observation
                    r_qpos = [self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] for j in self.joint_names]
                    r_qvel = [self.data.qvel[self.model.jnt_dofadr[self.model.joint(j).id]] for j in self.joint_names]
                    cur_obj_pos = self.data.qpos[obj_idx:obj_idx+3].copy()
                    cur_obj_quat = self.data.qpos[obj_idx+3:obj_idx+7].copy()
                    phys_params = self.get_physical_params(curr_obj_id)
                    
                    obs_raw = np.concatenate([r_qpos, r_qvel, cur_obj_pos, cur_obj_quat, target_quat, phys_params]).astype(np.float32)
                    obs_norm = (obs_raw - self.obs_mean) / self.obs_std
                    
                    obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
                        
                    current_qpos = np.array(r_qpos)
                    delta_q = np.clip(action - current_qpos, -0.15, 0.15)
                    safe_action = current_qpos + delta_q
                    
                    for i, aid in enumerate(self.actuator_ids):
                        if aid != -1:
                            self.data.ctrl[aid] = safe_action[i]
                            
                    for _ in range(10):  # 10 substeps * 0.002s = 0.02s per step
                        mujoco.mj_step(self.model, self.data)
                        
                    d_theta = compute_angular_distance(cur_obj_quat, target_quat)
                    final_err_deg = float(np.degrees(d_theta))
                    
                    if d_theta < 0.28:  # 16 deg tolerance
                        is_success = True
                    if cur_obj_pos[2] < 0.02 or np.linalg.norm(cur_obj_pos[:2]) > 0.20:
                        is_dropped = True
                        break

                    if viewer is not None:
                        draw_target_arrow(viewer, target_quat, cur_obj_pos)
                        viewer.sync()
                        elapsed = time.time() - step_start
                        if 0.02 - elapsed > 0:
                            time.sleep(0.02 - elapsed)

                errors_deg.append(final_err_deg)
                episode_lengths.append(step)
                if is_success:
                    successes += 1
                if is_dropped:
                    drops += 1
                    
                status_str = "SUCCESS" if is_success else ("DROPPED" if is_dropped else "TIME_LIMIT")
                print(f"  Finished: Steps: {step:3d}/{max_steps} | "
                      f"Final Error: {final_err_deg:5.2f}° | "
                      f"Status: {status_str}")

                if ep < num_episodes:
                    if viewer is not None and viewer.is_running():
                        delay_start = time.time()
                        while time.time() - delay_start < inter_episode_delay and viewer.is_running():
                            draw_target_arrow(viewer, target_quat, cur_obj_pos)
                            viewer.sync()
                            time.sleep(0.02)
                    else:
                        time.sleep(inter_episode_delay)

        finally:
            if viewer is not None:
                viewer.close()

        if episode_lengths:
            print("\n" + "="*65)
            print(f"BC EVALUATION SUMMARY ({len(episode_lengths)} Episodes Completed):")
            print(f"Success Rate:       {successes / len(episode_lengths) * 100:5.1f}% ({successes}/{len(episode_lengths)})")
            print(f"Drop Rate:          {drops / len(episode_lengths) * 100:5.1f}% ({drops}/{len(episode_lengths)})")
            print(f"Mean Steps:         {np.mean(episode_lengths):.1f} steps ({np.mean(episode_lengths)*0.02:.2f}s)")
            print(f"Mean Angular Error: {np.mean(errors_deg):.2f}° +/- {np.std(errors_deg):.2f}°")
            print("="*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Behavioral Cloning Policy with Visual Target Arrow & Quadratic Horizon")
    parser.add_argument("--checkpoint", type=str, default="in_hand_manipulation/bc_actor_checkpoint.pth", help="Path to BC checkpoint")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml", help="Path to scene XML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_goal_angle", type=float, default=0.80, help="Max target rotation angle in radians (default: 0.80 rad ~= 45.8 deg)")
    parser.add_argument("--object_id", type=int, default=0, help="Target object: 0=random, 1=cylinder, 2=cube, 3=cuboid (default: 0)")
    parser.add_argument("--delay", type=float, default=1.0, help="Inter-episode delay in seconds (default: 1.0s)")
    parser.add_argument("--no_render", action="store_true", help="Disable 3D viewer for fast headless benchmarking")
    args = parser.parse_args()
    
    evaluator = BCEvaluator(
        checkpoint_path=args.checkpoint,
        scene_xml_path=args.scene_xml
    )
    evaluator.run_evaluation(
        num_episodes=args.episodes,
        render=not args.no_render,
        max_goal_angle=args.max_goal_angle,
        object_id=args.object_id if args.object_id > 0 else None,
        inter_episode_delay=args.delay,
    )
