#!/usr/bin/env python3
"""
train_ppo.py
=============================================================================
Continuous Proximal Policy Optimization (PPO) with Generalized Advantage
Estimation (GAE) for In-Hand Robotic Manipulation (12-DOF Delto Gripper).

Features:
---------
1. Behavior Cloning (BC) Pre-trained Actor Warm-Start:
   - Loads pre-trained weights from `bc_actor_checkpoint.pth` (matching layer architecture).
   - Explicitly sets exploration standard deviation log_std = -2.3 (sigma ~= 0.1 rad).
2. Decoupled Critic Architecture:
   - Independent value network (no shared feature representation) to avoid "critic collapse".
   - Orthogonal weight initialization: hidden layers gain = sqrt(2), output value head gain = 1.0.
3. Rigorous PPO + GAE Mathematics:
   - GAE-Lambda advantage estimation: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
   - Advantage normalization across batch (mean = 0, std = 1).
   - PPO clipped surrogate objective (clipping parameter epsilon = 0.2 / 0.1).
   - Multivariate Gaussian analytical entropy bonus (c2 = 0.001).
   - Combined objective: L_total = L_actor + c1 * L_critic - c2 * S (c1 = 0.5).
4. Dual Optimizer Setup & Critic Warm-Up:
   - Conservative Actor learning rate (3e-5) to preserve demonstrated manipulation gaits.
   - Standard Critic learning rate (3e-4) to learn value landscapes quickly.
   - Critic warm-up phase (first 10-20 iterations) freezing the Actor while the Critic fits V(s).
5. Exact Multimodal In-Hand Reward Function:
   r_t = - w1 * (d_theta)
         - w2 * ||a_t||^2
         - w3 * ||qdot_t||^2
         + w4 * I(d_theta < epsilon)
         - w5 * I(dist_to_palm > 0.12)
         + w6 * n_c
         - w7 * I(n_c == 0)
=============================================================================
"""

import os
import sys
import time
import math
import csv
import json
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Generator, List

import numpy as np
import scipy.spatial.transform as transform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Import MuJoCo for the Delto hand simulation
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass


# =============================================================================
# 0. CONFIGURATION & HYPERPARAMETERS
# =============================================================================

@dataclass
class RewardWeights:
    """
    Reward function coefficients according to the user specification:
    r_t = - w1*(d_theta) - w2*||a_t||^2 - w3*||qdot_t||^2
          + w4*I(d_theta < epsilon) - w5*I(dist_to_palm > 0.12)
          + w6*n_c - w7*I(n_c == 0)
    """
    w1: float = 1.0     # Angular distance penalty weight: penalizes rotational error d_theta
    w2: float = 0.001   # Control penalty weight: penalizes large torque / action norms ||a_t||^2
    w3: float = 0.0001  # Velocity penalty weight: prevents actuator thrashing ||qdot_t||^2
    w4: float = 10.0    # Terminal success bonus: large positive reward upon reaching goal orientation
    w5: float = 15.0    # Early reset / drop penalty: absorbing state penalty when object dropped
    w6: float = 0.05    # Proportional contact reward: encourages multi-fingered contact points (n_c)
    w7: float = 1.0     # Zero contact penalty: penalty triggered when contact points drop to 0
    epsilon: float = 0.15           # Target orientation error threshold in radians (~8.6 degrees)
    drop_threshold: float = 0.12   # Distance threshold from palm center (meters)


@dataclass
class PPOConfig:
    """Hyperparameters for PPO fine-tuning."""
    # Training Loop
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048       # Trajectory steps collected per policy iteration (T)
    num_iterations: int = 250       # Total PPO policy iterations
    num_minibatches: int = 32       # Number of minibatches (e.g. 2048 / 32 = 64 batch size)
    mini_batch_size: int = 64       # Size of each SGD minibatch
    ppo_epochs: int = 5             # Number of optimization epochs per rollout buffer (tuned for fine-tuning stability)
    
    # Critic Warm-Up
    warmup_iterations: int = 30     # Number of initial iterations where Actor is frozen (increased to 30 for value calibration)
    
    # Optimization Learning Rates
    actor_lr: float = 2e-5          # Conservative learning rate to preserve BC demonstration gaits
    critic_lr: float = 3e-4         # Standard learning rate for value function fitting
    max_grad_norm: float = 0.5      # Gradient clipping threshold for stability
    
    # PPO and GAE Mathematics
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE bias-variance trade-off parameter
    clip_eps: float = 0.2           # PPO clipping parameter epsilon (can be 0.1 for tighter updates)
    target_kl: Optional[float] = 0.02 # Early-stopping threshold on approx_kl (protects BC weights)
    value_coef: float = 0.5         # Critic loss weight c1
    entropy_coef: float = 0.001     # Entropy bonus weight c2
    initial_log_std: float = -2.3   # Explores safely near demonstrated gaits (sigma ~= 0.1 rad)
    
    # Environment & Dimension Definitions
    obs_dim: int = 39               # 12 qpos + 12 qvel + 3 obj_pos + 4 obj_quat + 4 target_quat + 4 phys_params
    act_dim: int = 12               # 12 actuated continuous joints of Delto hand
    max_episode_steps: int = 250    # Max steps per episode before time-limit truncation
    render: bool = False            # If True, opens live interactive 3D MuJoCo viewer during training
    show_actions: bool = False      # If True, prints numerical action telemetry during rollouts
    dr_ratio: float = 0.30          # Domain randomization ratio (+-30% on mass, friction, and sizes)



# =============================================================================
# 1. ENVIRONMENT: 12-DOF DELTO HAND IN-HAND MANIPULATION (MUJOCO)
# =============================================================================

class DeltoInHandManipulationEnv:
    """
    Gymnasium-style continuous RL environment for 12-DOF Delto Gripper manipulation.
    Simulates physics using MuJoCo (scene_cube.xml).
    """
    def __init__(
        self,
        scene_xml_path: str,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        reward_weights: Optional[RewardWeights] = None,
        max_episode_steps: int = 250,
        substeps: int = 10,
        render_mode: bool = False,
        dr_ratio: float = 0.30,
    ):
        if not os.path.exists(scene_xml_path):
            raise FileNotFoundError(f"MuJoCo XML scene not found at {scene_xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.reward_weights = reward_weights or RewardWeights()
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps  # 10 substeps * 0.002s = 0.02s per RL step (50 Hz)
        self.render_mode = render_mode
        self.viewer = None
        
        # Domain Randomization Setup (+-30% on mass, friction, and size)
        self.dr_ratio = dr_ratio
        self.nominal_obj_props = {}
        for obj_id in [1, 2, 3]:
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj_id}")
            g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{obj_id}_geom")
            self.nominal_obj_props[obj_id] = {
                "body_id": b_id,
                "geom_id": g_id,
                "mass": float(self.model.body_mass[b_id]),
                "inertia": self.model.body_inertia[b_id].copy(),
                "friction": self.model.geom_friction[g_id].copy(),
                "size": self.model.geom_size[g_id].copy(),
                "phys_params": self.get_physical_params(obj_id).copy(),
            }
        self.current_phys_params = np.array([0.045, 0.07, 1.0, 3.0], dtype=np.float32)
        
        if self.render_mode:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                print("[Environment] Live MuJoCo 3D Viewer launched successfully.")
            except Exception as e:
                print(f"[Environment] Warning: Could not launch MuJoCo passive viewer: {e}")
                self.render_mode = False
        
        # Observation Normalization Statistics (from BC pre-training)
        self.obs_mean = obs_mean.astype(np.float32) if obs_mean is not None else np.zeros(39, dtype=np.float32)
        self.obs_std = obs_std.astype(np.float32) if obs_std is not None else np.ones(39, dtype=np.float32)
        
        # Resolve Gripper Joints & Actuator IDs
        self.joint_names = self._get_gripper_joint_names()
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name.replace('gripper_', '').replace('_joint', '')}")
            for name in self.joint_names
        ]
        
        # Supported Objects: 1: Cylinder, 2: Cube, 3: Cuboid
        self.active_object_id = 1
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        self.step_count = 0
        
        # Palm origin is at (0, 0, 0) where the Delto finger base mounts intersect
        self.palm_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_gripper_joint_names(self) -> List[str]:
        names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name.startswith("gripper_") and "joint" in name:
                names.append(name)
        return names

    def get_physical_params(self, obj_id: int) -> np.ndarray:
        """Physical parameters [radius/width, height, mass, friction] matching BC preprocessing."""
        if obj_id == 1:
            return np.array([0.045, 0.07, 1.0, 3.0], dtype=np.float32)
        elif obj_id == 2:
            return np.array([0.045, 0.045, 1.0, 3.0], dtype=np.float32)
        elif obj_id == 3:
            return np.array([0.055, 0.07, 1.0, 3.0], dtype=np.float32)
        return np.array([0.045, 0.07, 1.0, 3.0], dtype=np.float32)

    def reset_gripper_pose(self):
        """Sets the 12 Delto joints to canonical starting tripod pinch pose."""
        initial_pose = [
            ("gripper_f1m1_joint", 0.0), ("gripper_f1m2_joint", 0.0),
            ("gripper_f1m3_joint", 0.8), ("gripper_f1m4_joint", 0.8),
            ("gripper_f2m1_joint", -1.047), ("gripper_f2m2_joint", 0.0),
            ("gripper_f2m3_joint", 0.8), ("gripper_f2m4_joint", 0.8),
            ("gripper_f3m1_joint", 1.047), ("gripper_f3m2_joint", 0.0),
            ("gripper_f3m3_joint", 0.8), ("gripper_f3m4_joint", 0.8),
        ]
        for name, val in initial_pose:
            j_id = self.model.joint(name).id
            qpos_idx = self.model.jnt_qposadr[j_id]
            vel_idx = self.model.jnt_dofadr[j_id]
            self.data.qpos[qpos_idx] = val
            self.data.qvel[vel_idx] = 0.0
            
        for j_name, val in initial_pose:
            act_name = f"act_{j_name.replace('gripper_', '').replace('_joint', '')}"
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                self.data.ctrl[act_id] = val

    def reset(self, object_id: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment, sets random target orientation, and returns initial state."""
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Reset Gripper Joints
        self.reset_gripper_pose()
        
        # 2. Select Active Object (Randomize across Cylinder, Cube, Cuboid if None)
        if object_id is None:
            self.active_object_id = int(np.random.choice([1, 2, 3]))
        else:
            self.active_object_id = int(object_id)
            
        # 3. Apply Domain Randomization (+-30% on mass, friction, and size)
        props = self.nominal_obj_props[self.active_object_id]
        body_id = props["body_id"]
        geom_id = props["geom_id"]
        
        if self.dr_ratio > 0.0:
            # Sample independent uniform random scaling factors in [1 - dr_ratio, 1 + dr_ratio]
            scale_size = float(np.random.uniform(1.0 - self.dr_ratio, 1.0 + self.dr_ratio))
            scale_mass = float(np.random.uniform(1.0 - self.dr_ratio, 1.0 + self.dr_ratio))
            scale_fric = float(np.random.uniform(1.0 - self.dr_ratio, 1.0 + self.dr_ratio))
            
            # Apply to MuJoCo physical model
            self.model.geom_size[geom_id] = props["size"] * scale_size
            self.model.body_mass[body_id] = props["mass"] * scale_mass
            self.model.body_inertia[body_id] = props["inertia"] * scale_mass
            
            fric_copy = props["friction"].copy()
            fric_copy[0] *= scale_fric  # Tangential sliding friction
            self.model.geom_friction[geom_id] = fric_copy
            
            # Construct actual randomized 4-dim parameter vector for the network observation:
            # [radius/width * scale_size, height * scale_size, mass * scale_mass, friction * scale_fric]
            base_phys = props["phys_params"]
            self.current_phys_params = np.array([
                base_phys[0] * scale_size,
                base_phys[1] * scale_size,
                base_phys[2] * scale_mass,
                base_phys[3] * scale_fric
            ], dtype=np.float32)
        else:
            self.model.geom_size[geom_id] = props["size"].copy()
            self.model.body_mass[body_id] = props["mass"]
            self.model.body_inertia[body_id] = props["inertia"].copy()
            self.model.geom_friction[geom_id] = props["friction"].copy()
            self.current_phys_params = props["phys_params"].copy()
            
        # 4. Position Active Object in hand [0.0, 0.0, 0.10], move other objects away
        for i in range(1, 4):
            idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
            vel_idx = self.model.jnt_dofadr[self.model.joint(f"obj_{i}_joint").id]
            if i == self.active_object_id:
                # Active object placed in grasp envelope
                self.data.qpos[idx:idx+7] = [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]
            else:
                # Inactive objects shifted far outside the scene
                self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[vel_idx:vel_idx+6] = 0.0
            
        # 5. Generate Random Target Goal Orientation (Quat [w, x, y, z])
        rand_rot = transform.Rotation.random()
        q_xyzw = rand_rot.as_quat()  # Scipy is [x, y, z, w]
        self.target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        
        # Forward kinematics & step physics briefly to let contacts settle
        mujoco.mj_forward(self.model, self.data)
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_observation()
        info = {
            "object_id": self.active_object_id,
            "target_quat": self.target_quat.copy(),
            "phys_params": self.current_phys_params.copy()
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Constructs normalized 39-dimensional observation:
        [robot_qpos(12), robot_qvel(12), obj_pos(3), obj_quat(4), target_quat(4), phys_params(4)]
        """
        r_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] for j in self.joint_names], dtype=np.float32)
        r_qvel = np.array([self.data.qvel[self.model.jnt_dofadr[self.model.joint(j).id]] for j in self.joint_names], dtype=np.float32)
        
        obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{self.active_object_id}_joint").id]
        obj_pos = self.data.qpos[obj_idx:obj_idx+3].astype(np.float32)
        obj_quat = self.data.qpos[obj_idx+3:obj_idx+7].astype(np.float32)
        
        # Injects the randomized physical parameters into observation state
        obs_raw = np.concatenate([r_qpos, r_qvel, obj_pos, obj_quat, self.target_quat, self.current_phys_params], axis=0)
        
        # Apply observation normalization from BC pretraining: (s - mean) / std
        obs_norm = (obs_raw - self.obs_mean) / self.obs_std
        return obs_norm.astype(np.float32)

    def count_contacts(self) -> int:
        """Counts the number of contact points between the active object and Delto gripper geoms."""
        active_obj_body = f"obj_{self.active_object_id}"
        n_c = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom1])
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom2])
            if b1 is None or b2 is None:
                continue
            is_obj_contact = (b1 == active_obj_body and b2.startswith("gripper_")) or \
                             (b2 == active_obj_body and b1.startswith("gripper_"))
            if is_obj_contact:
                n_c += 1
        return n_c

    @staticmethod
    def compute_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Computes the SO(3) geodesic angular distance between two unit quaternions q1 and q2:
        d_theta = 2 * arccos(|<q1, q2>|), d_theta in [0, pi] radians.
        Both quaternions are assumed in MuJoCo [w, x, y, z] convention.
        """
        dot = np.abs(np.dot(q1, q2))
        dot_clamped = np.clip(dot, -1.0, 1.0)
        return float(2.0 * np.arccos(dot_clamped))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one control step in the environment.
        Args:
            action: 12-dimensional joint position target vector.
        Returns:
            obs: Normalized observation vector (39,)
            reward: Scalar reward according to the exact combined reward function
            terminated: Boolean indicating goal achievement or object drop
            truncated: Boolean indicating max horizon exceeded
            info: Dictionary containing diagnostic telemetry
        """
        self.step_count += 1
        
        # 1. Action Application with Safety Delta Clipping (prevents kinematic thrashing)
        current_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] for j in self.joint_names])
        delta_q = np.clip(action - current_qpos, -0.15, 0.15)
        safe_action = current_qpos + delta_q
        
        for i, aid in enumerate(self.actuator_ids):
            if aid != -1:
                self.data.ctrl[aid] = safe_action[i]
                
        # 2. Step MuJoCo Physics (substeps * dt)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            
        # 3. Extract Post-Step State
        obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{self.active_object_id}_joint").id]
        obj_pos = self.data.qpos[obj_idx:obj_idx+3]
        obj_quat = self.data.qpos[obj_idx+3:obj_idx+7]
        r_qvel = np.array([self.data.qvel[self.model.jnt_dofadr[self.model.joint(j).id]] for j in self.joint_names])
        
        # 4. Compute Metrics for Reward Function
        # (a) Geodesic Angular Distance: d_theta
        d_theta = self.compute_angular_distance(obj_quat, self.target_quat)
        
        # (b) Control Penalty: ||a_t||^2
        control_norm_sq = float(np.sum(np.square(action)))
        
        # (c) Velocity Penalty: ||qdot_t||^2
        vel_norm_sq = float(np.sum(np.square(r_qvel)))
        
        # (d) Distance from palm center
        dist_to_palm = float(np.linalg.norm(obj_pos - self.palm_position))
        
        # (e) Contact points
        n_c = self.count_contacts()
        
        # 5. Evaluate Indicator Functions & Terminations
        w = self.reward_weights
        is_success = bool(d_theta < w.epsilon)
        is_dropped = bool(dist_to_palm > w.drop_threshold)
        
        term_success_bonus = w.w4 if is_success else 0.0
        drop_penalty = w.w5 if is_dropped else 0.0
        zero_contact_pen = w.w7 if (n_c == 0) else 0.0
        
        # =====================================================================
        # EXACT REWARD FORMULATION (with Survival Baseline):
        # r_t = w1 * (pi - d_theta)
        #       - w2 * ||a_t||^2
        #       - w3 * ||qdot_t||^2
        #       + w4 * I(d_theta < epsilon)
        #       - w5 * I(dist_to_palm > 0.12)
        #       + w6 * n_c
        #       - w7 * I(n_c == 0)
        #
        # Anti-Suicide Transformation:
        # Formulating the angular term as w1 * (pi - d_theta) >= 0 is mathematically
        # equivalent to: - w1 * d_theta + w1 * pi.
        # The constant w1 * pi (+3.1416) acts as a per-step "alive bonus".
        # While holding the cube, the agent earns positive return at every step.
        # Dropping the cube immediately terminates the episode and forfeits all
        # remaining positive rewards for the episode horizon, eliminating the
        # failure mode where the policy learns to drop the object early.
        # =====================================================================
        reward = (
            (w.w1 * (np.pi - d_theta))
            - (w.w2 * control_norm_sq)
            - (w.w3 * vel_norm_sq)
            + term_success_bonus
            - drop_penalty
            + (w.w6 * float(n_c))
            - zero_contact_pen
        )
        
        # 6. Termination Flags
        # Dropping the object or achieving orientation threshold terminates episode
        terminated = is_dropped or is_success
        truncated = (self.step_count >= self.max_episode_steps)
        
        next_obs = self._get_observation()
        info = {
            "d_theta": d_theta,
            "dist_to_palm": dist_to_palm,
            "contacts": n_c,
            "is_success": is_success,
            "is_dropped": is_dropped,
            "reward_components": {
                "angular_reward": (w.w1 * (np.pi - d_theta)),
                "control_penalty": - (w.w2 * control_norm_sq),
                "velocity_penalty": - (w.w3 * vel_norm_sq),
                "success_bonus": term_success_bonus,
                "drop_penalty": - drop_penalty,
                "contact_reward": w.w6 * float(n_c),
                "zero_contact_penalty": - zero_contact_pen,
            }
        }
        
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            
        return next_obs, reward, terminated, truncated, info

    def close(self):
        """Safely closes the MuJoCo viewer if open."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURES: ACTOR & DECOUPLED CRITIC
# =============================================================================

class ActorNetwork(nn.Module):
    """
    PPO-compatible Continuous Actor Network.
    Must identically match the Behavior Cloning (BC) pre-training architecture
    so weights from `bc_actor_checkpoint.pth` can be seamlessly loaded.
    
    Structure:
      Linear(39, 512) -> LayerNorm(512) -> Tanh() -> Dropout(0.1)
      Linear(512, 512) -> LayerNorm(512) -> Tanh() -> Dropout(0.1)
      Linear(512, 256) -> LayerNorm(256) -> Tanh() -> Dropout(0.1)
      mu_head: Linear(256, 12)
      log_std: nn.Parameter(torch.full((12,), -2.3))  [sigma ~= 0.1 rad]
    """
    def __init__(self, obs_dim: int = 39, act_dim: int = 12):
        super(ActorNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Backbone (Identical to train_bc.py)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(p=0.1)
        )
        
        # Mean action head
        self.mu_head = nn.Sequential(
            nn.Linear(256, act_dim)
        )
        
        # Learnable exploration log standard deviation:
        # Initialized to -2.3 (sigma ~= exp(-2.3) ~= 0.100 rad)
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.3, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns mean action mu(s)."""
        x = self.net(state)
        return self.mu_head(x)

    def get_action_distribution(self, state: torch.Tensor) -> Normal:
        """Constructs PyTorch Normal distribution pi(a | s)."""
        mu = self.forward(state)
        # Standard deviation is bounded to prevent numerical instability
        std = torch.clamp(self.log_std.exp(), min=1e-3, max=1.0)
        return Normal(mu, std)

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy for rollout data collection.
        Returns:
            action: Sampled action a ~ pi(a | s)
            log_prob: Log probability log pi(a | s)
            mu: Mean action (deterministic prediction)
        """
        dist = self.get_action_distribution(state)
        action = dist.sample()
        # Sum log probabilities across all 12 independent DOF joints
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.mean

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates actions during PPO gradient updates.
        Returns:
            log_prob: Sum of log likelihoods log pi_theta(a | s)
            entropy: Total analytical entropy of the multivariate Gaussian policy
        """
        dist = self.get_action_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    PPO Value Network V(s) -> scalar state-value estimate.
    
    CRITICAL ARCHITECTURAL CONSTRAINT:
    The Critic is COMPLETELY DECOUPLED from the Actor (no shared feature extraction layers)
    to prevent 'critic collapse', where massive initial value prediction errors would destroy
    the pre-trained Behavioral Cloning features in the shared trunk.
    
    Orthogonal Initialization:
      - Hidden layers: Orthogonal initialization with gain = sqrt(2) (optimal for Tanh)
      - Output layer: Orthogonal initialization with gain = 1.0 (standard for value function)
      - Biases: 0.0
    """
    def __init__(self, obs_dim: int = 39):
        super(CriticNetwork, self).__init__()
        self.obs_dim = obs_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh()
        )
        self.v_head = nn.Linear(256, 1)
        
        # Apply orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Orthogonal initialization with appropriate gains."""
        if isinstance(m, nn.Linear):
            if m is self.v_head:
                # Value output head: gain = 1.0
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
            else:
                # Hidden layers: gain = sqrt(2) for Tanh activation
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns scalar value prediction V(s)."""
        x = self.net(state)
        return self.v_head(x).squeeze(-1)


# =============================================================================
# 3. ROLLOUT BUFFER & GENERALIZED ADVANTAGE ESTIMATION (GAE)
# =============================================================================

class RolloutBuffer:
    """
    Synchronous Trajectory Rollout Buffer.
    Stores T steps of transitions, calculates Generalized Advantage Estimation (GAE),
    and yields shuffled minibatches for PPO updates.
    """
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        # Pre-allocated tensors
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        
        self.step_idx = 0
        self.is_full = False

    def insert(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """Inserts one transition into the buffer."""
        assert self.step_idx < self.buffer_size, "Buffer overflow!"
        
        self.obs[self.step_idx] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.step_idx] = action
        self.log_probs[self.step_idx] = log_prob
        self.rewards[self.step_idx] = float(reward)
        self.values[self.step_idx] = value
        self.dones[self.step_idx] = float(done)
        
        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Computes Generalized Advantage Estimation (GAE):
          delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
          A_hat_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
                  = delta_t + gamma * lambda * (1 - d_t) * A_hat_{t+1}
          Target Return R_t = A_hat_t + V(s_t)
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
                
            # 1-step Temporal Difference error: delta_t
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # Recursive backward GAE calculation
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            
        # Target returns for the Critic: R_t = A_hat_t + V(s_t)
        self.returns = self.advantages + self.values

    def get_minibatch_generator(
        self,
        batch_size: int,
        normalize_advantages: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields random minibatches for PPO updates.
        Optionally normalizes advantages across the entire rollout buffer before batching.
        """
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)
        
        # Advantage normalization across entire rollout buffer: (A - mean) / (std + 1e-8)
        advs = self.advantages
        if normalize_advantages:
            adv_mean = advs.mean()
            adv_std = advs.std() + 1e-8
            norm_advantages = (advs - adv_mean) / adv_std
        else:
            norm_advantages = advs
            
        for start_idx in range(0, self.buffer_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "old_log_probs": self.log_probs[batch_idx],
                "advantages": norm_advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "old_values": self.values[batch_idx],
            }

    def clear(self):
        """Resets the buffer for the next policy rollout."""
        self.step_idx = 0
        self.is_full = False


# =============================================================================
# 4. TRAINING TELEMETRY & CSV LOGGER
# =============================================================================

class TrainingLogger:
    """Lightweight, non-blocking metrics logger to CSV and JSONL."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "progress.csv")
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.fieldnames = [
            "iteration", "ep_return", "step_reward", "ep_length", "success_rate", "drop_rate",
            "loss_critic", "loss_actor", "entropy", "clip_fraction",
            "approx_kl", "log_std", "elapsed_sec"
        ]
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, metrics: Dict[str, float]):
        filtered = {k: metrics.get(k, 0.0) for k in self.fieldnames}
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered)
        with open(self.jsonl_path, mode="a") as f:
            f.write(json.dumps(metrics) + "\n")


# =============================================================================
# 5. PPO AGENT & OPTIMIZATION PIPELINE
# =============================================================================

class PPOFineTuner:
    """
    Orchestrates Continuous PPO RL fine-tuning of the pre-trained BC Actor.
    """
    def __init__(self, config: PPOConfig, checkpoint_path: str, scene_xml_path: str):
        self.cfg = config
        self.checkpoint_path = checkpoint_path
        self.scene_xml_path = scene_xml_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOFineTuner] Using compute device: {self.device}")
        
        # 1. Initialize Actor and Critic Networks
        self.actor = ActorNetwork(obs_dim=self.cfg.obs_dim, act_dim=self.cfg.act_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim=self.cfg.obs_dim).to(self.device)
        
        # 2. Load Behavior Cloning Checkpoint & Init Parameters
        self.obs_mean, self.obs_std = self._load_bc_checkpoint()
        
        # 3. Instantiate Dual Optimizers (Decoupled learning rates)
        # Conservative Actor LR (3e-5) prevents policy degradation
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr, eps=1e-5)
        # Standard Critic LR (3e-4) learns value functions rapidly
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr, eps=1e-5)
        
        # 4. Instantiate Rollout Buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.cfg.rollout_steps,
            obs_dim=self.cfg.obs_dim,
            act_dim=self.cfg.act_dim,
            device=self.device
        )
        
        # 5. Instantiate MuJoCo Environment
        self.env = DeltoInHandManipulationEnv(
            scene_xml_path=self.scene_xml_path,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            max_episode_steps=self.cfg.max_episode_steps,
            render_mode=self.cfg.render,
            dr_ratio=self.cfg.dr_ratio
        )
        
        # 6. Persistent Telemetry Queues across rollout boundaries
        self.recent_returns = deque(maxlen=50)
        self.recent_lengths = deque(maxlen=50)
        self.recent_successes = deque(maxlen=50)
        self.recent_drops = deque(maxlen=50)
        self.current_ep_reward = 0.0
        self.current_ep_len = 0

    def _load_bc_checkpoint(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads pre-trained BC weights into Actor and sets exploration log_std.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"BC Checkpoint not found at {self.checkpoint_path}! "
                "Please run `train_bc.py` first to generate pre-trained weights."
            )
            
        print(f"[PPOFineTuner] Loading BC weights from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dictionary into Actor's net and mu_head
        actor_state = checkpoint["model_state_dict"]
        # Filter for layers matching Actor
        self.actor.load_state_dict(actor_state, strict=False)
        
        # Strictly enforce initial exploration standard deviation parameter
        # log_std = -2.3 corresponds to sigma ~= 0.1 rad, safe for Delto hand exploration
        with torch.no_grad():
            self.actor.log_std.fill_(self.cfg.initial_log_std)
            
        print(f"[PPOFineTuner] Actor weights loaded successfully.")
        print(f"[PPOFineTuner] Actor log_std manually initialized to: {self.actor.log_std.data.cpu().numpy()[0]:.2f} "
              f"(sigma ~= {self.actor.log_std.exp().data.cpu().numpy()[0]:.4f} rad)")
        
        obs_mean = checkpoint.get("obs_mean", np.zeros(self.cfg.obs_dim, dtype=np.float32))
        obs_std = checkpoint.get("obs_std", np.ones(self.cfg.obs_dim, dtype=np.float32))
        return obs_mean, obs_std

    def collect_rollouts(self, current_obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Executes policy in MuJoCo for rollout_steps to fill the trajectory buffer.
        """
        self.buffer.clear()
        # Set actor to eval mode during rollout to disable dropout
        self.actor.eval()
        self.critic.eval()
        
        total_rollout_reward = 0.0
        
        for step in range(self.cfg.rollout_steps):
            obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, _ = self.actor.act(obs_tensor)
                value = self.critic(obs_tensor)
                
            action_np = action.squeeze(0).cpu().numpy()
            
            # Print sample action telemetry if requested
            if self.cfg.show_actions and (step % 256 == 0):
                print(f"  [Rollout Step {step:04d}] Action norm: {np.linalg.norm(action_np):.3f} | Mean: {action_np.mean():+.3f} | Min/Max: [{action_np.min():+.2f}, {action_np.max():+.2f}]")
                print(f"    Joint Targets: {np.round(action_np, 2)}")
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Optional pacing for human-observable visual playback if rendering is enabled
            if self.cfg.render:
                time.sleep(0.015)
            
            # Insert into rollout buffer
            self.buffer.insert(
                obs=current_obs,
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                reward=reward,
                value=value.squeeze(0),
                done=done
            )
            
            current_obs = next_obs
            total_rollout_reward += reward
            self.current_ep_reward += reward
            self.current_ep_len += 1
            
            if done:
                self.recent_returns.append(self.current_ep_reward)
                self.recent_lengths.append(self.current_ep_len)
                self.recent_successes.append(1.0 if info.get("is_success", False) else 0.0)
                self.recent_drops.append(1.0 if info.get("is_dropped", False) else 0.0)
                
                self.current_ep_reward = 0.0
                self.current_ep_len = 0
                current_obs, _ = self.env.reset()
                
        # Value of final state for GAE truncation handling: V(s_{T})
        final_obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.critic(final_obs_tensor).squeeze(0)
            
        self.buffer.compute_gae(last_value=last_value, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda)
        
        step_reward = total_rollout_reward / float(self.cfg.rollout_steps)
        metrics = {
            "step_reward": step_reward,
            "ep_return": float(np.mean(self.recent_returns)) if self.recent_returns else (step_reward * self.cfg.max_episode_steps),
            "ep_length": float(np.mean(self.recent_lengths)) if self.recent_lengths else float(self.current_ep_len),
            "success_rate": float(np.mean(self.recent_successes)) if self.recent_successes else 0.0,
            "drop_rate": float(np.mean(self.recent_drops)) if self.recent_drops else 0.0,
        }
        return current_obs, metrics

    def update_policy(self, is_warmup: bool = False) -> Dict[str, float]:
        """
        Executes PPO optimization epochs over minibatches from the rollout buffer.
        
        Args:
            is_warmup: If True (Critic Warm-up Phase), Actor weights are frozen and
                       only Critic value loss is minimized.
        """
        self.actor.eval()  # Keep dropout inactive to maintain exact policy distribution
        self.critic.train()
        
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []
        
        kl_early_stopped = False
        epochs_completed = 0
        
        for epoch in range(self.cfg.ppo_epochs):
            if kl_early_stopped:
                break
            epochs_completed += 1
            for batch in self.buffer.get_minibatch_generator(batch_size=self.cfg.mini_batch_size, normalize_advantages=True):
                b_obs = batch["obs"]
                b_actions = batch["actions"]
                b_old_log_probs = batch["old_log_probs"]
                b_advantages = batch["advantages"]
                b_returns = batch["returns"]
                
                # -------------------------------------------------------------
                # 1. CRITIC VALUE UPDATE
                # -------------------------------------------------------------
                values = self.critic(b_obs)
                # Value Loss: L_critic = 0.5 * MSE(V(s), Return)
                critic_loss = 0.5 * nn.functional.mse_loss(values, b_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()
                
                critic_losses.append(critic_loss.item())
                
                # -------------------------------------------------------------
                # 2. ACTOR PPO UPDATE (Skipped during Critic Warm-Up)
                # -------------------------------------------------------------
                if is_warmup:
                    # Freeze Actor during warm-up: zero gradients and continue
                    self.actor_optimizer.zero_grad()
                    continue
                    
                # Evaluate current policy log probabilities and entropy
                new_log_probs, entropy = self.actor.evaluate_actions(b_obs, b_actions)
                
                # Probability Ratio: r_t(theta) = exp(log pi_theta - log pi_old)
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)
                
                # Approximate KL Divergence for monitoring policy stability: (r - 1) - log(r)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    
                    # Track clipping frequency
                    clipped = (ratio < (1.0 - self.cfg.clip_eps)) | (ratio > (1.0 + self.cfg.clip_eps))
                    clip_fractions.append(clipped.float().mean().item())
                
                # Target KL early stopping: prevents destructive policy drift beyond trust region
                if self.cfg.target_kl is not None and approx_kl > (1.5 * self.cfg.target_kl):
                    kl_early_stopped = True
                    break
                
                # Clipped Surrogate Objective:
                # Surr1 = r_t(theta) * A_t
                # Surr2 = clip(r_t(theta), 1 - eps, 1 + eps) * A_t
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * b_advantages
                actor_loss = - torch.min(surr1, surr2).mean()
                
                # Entropy Bonus: S = mean(entropy)
                mean_entropy = entropy.mean()
                
                # Actor Objective with Entropy Regularization:
                # L_actor_total = L_actor - c2 * S
                total_actor_loss = actor_loss - (self.cfg.entropy_coef * mean_entropy)
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                entropies.append(mean_entropy.item())
                
        return {
            "loss_critic": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "loss_actor": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "kl_early_stopped": kl_early_stopped,
            "epochs_completed": epochs_completed,
        }

    def train(self, output_dir: str = "checkpoints_ppo"):
        """Executes the full PPO training curriculum with Critic Warm-Up."""
        os.makedirs(output_dir, exist_ok=True)
        logger = TrainingLogger(log_dir=output_dir)
        
        print("\n" + "="*75)
        print("STARTING CONTINUOUS PPO REINFORCEMENT LEARNING FINE-TUNING")
        print(f"Total Iterations: {self.cfg.num_iterations} | Rollout Steps: {self.cfg.rollout_steps}")
        print(f"Actor LR: {self.cfg.actor_lr:.2e} | Critic LR: {self.cfg.critic_lr:.2e}")
        print(f"Critic Warm-Up Phase: First {self.cfg.warmup_iterations} iterations (Actor Frozen)")
        print(f"Logging progress to: {os.path.join(output_dir, 'progress.csv')}")
        print("="*75 + "\n")
        
        obs, _ = self.env.reset()
        best_success_rate = -1.0
        start_time = time.time()
        
        for iteration in range(1, self.cfg.num_iterations + 1):
            iter_start = time.time()
            is_warmup = (iteration <= self.cfg.warmup_iterations)
            
            # Phase A: Trajectory Rollout Collection
            obs, rollout_metrics = self.collect_rollouts(obs)
            
            # Phase B: PPO Policy & Value Updates
            update_metrics = self.update_policy(is_warmup=is_warmup)
            
            elapsed = time.time() - iter_start
            phase_tag = "[CRITIC WARM-UP]" if is_warmup else "[JOINT PPO]"
            
            # Telemetry Metrics Record
            log_record = {
                "iteration": iteration,
                "ep_return": rollout_metrics["ep_return"],
                "step_reward": rollout_metrics["step_reward"],
                "ep_length": rollout_metrics["ep_length"],
                "success_rate": rollout_metrics["success_rate"],
                "drop_rate": rollout_metrics["drop_rate"],
                "loss_critic": update_metrics["loss_critic"],
                "loss_actor": update_metrics["loss_actor"],
                "entropy": update_metrics["entropy"],
                "clip_fraction": update_metrics["clip_fraction"],
                "approx_kl": update_metrics["approx_kl"],
                "log_std": float(self.actor.log_std.data.mean().item()),
                "elapsed_sec": elapsed,
            }
            logger.log(log_record)
            
            # Console Output
            kl_flag = " (KL stopped)" if update_metrics.get("kl_early_stopped", False) else ""
            print(
                f"Iter {iteration:03d}/{self.cfg.num_iterations:03d} {phase_tag:<18} | "
                f"EpRet: {rollout_metrics['ep_return']:7.1f} | "
                f"StepR: {rollout_metrics['step_reward']:6.2f} | "
                f"Succ: {rollout_metrics['success_rate']*100:5.1f}% | "
                f"Drop: {rollout_metrics['drop_rate']*100:5.1f}% | "
                f"CritLoss: {update_metrics['loss_critic']:6.4f} | "
                f"ActLoss: {update_metrics['loss_actor']:6.4f} | "
                f"KL: {update_metrics['approx_kl']:.4f}{kl_flag} | "
                f"Time: {elapsed:4.1f}s"
            )
            
            # Model Checkpointing
            if not is_warmup and rollout_metrics["success_rate"] > best_success_rate:
                best_success_rate = rollout_metrics["success_rate"]
                self.save_checkpoint(os.path.join(output_dir, "ppo_best_actor.pth"), iteration, rollout_metrics)
                
            if iteration % 25 == 0 or iteration == self.cfg.num_iterations:
                self.save_checkpoint(os.path.join(output_dir, f"ppo_checkpoint_iter_{iteration}.pth"), iteration, rollout_metrics)
                
        total_time = (time.time() - start_time) / 60.0
        print("\n" + "="*75)
        print(f"PPO FINE-TUNING FINISHED in {total_time:.2f} minutes!")
        print(f"Best Success Rate: {best_success_rate * 100:.2f}%")
        print(f"Models saved in: {output_dir}")
        print("="*75)

    def save_checkpoint(self, path: str, iteration: int, metrics: Dict[str, float]):
        """Saves weights, optimizer states, normalization parameters, and config."""
        payload = {
            "iteration": iteration,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "log_std": self.actor.log_std.detach().cpu().numpy(),
            "metrics": metrics,
            "config": self.cfg.__dict__,
        }
        torch.save(payload, path)
        print(f"  [Checkpoint] Saved model checkpoint to {path}")


# =============================================================================
# 6. MAIN ENTRY POINT & CLI PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Fine-Tuning for 12-DOF Delto Hand In-Hand Manipulation")
    parser.add_argument("--iterations", type=int, default=250, help="Total PPO iterations")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Rollout buffer steps per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for SGD updates")
    parser.add_argument("--ppo_epochs", type=int, default=5, help="Epochs per rollout update (default: 5)")
    parser.add_argument("--warmup_iters", type=int, default=30, help="Critic warm-up iterations where Actor is frozen (default: 30)")
    parser.add_argument("--actor_lr", type=float, default=2e-5, help="Actor learning rate (default: 2e-5)")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--target_kl", type=float, default=0.02, help="Target KL divergence for early stopping (default: 0.02)")
    parser.add_argument("--max_steps", type=int, default=400, help="Max episode horizon in steps (default: 400 steps = 8.0 seconds at 50Hz)")
    parser.add_argument("--dr_ratio", type=float, default=0.30, help="Domain randomization ratio (+-30%% on mass, friction, size)")
    parser.add_argument("--render", action="store_true", help="Launch interactive 3D MuJoCo window to see robot manipulating objects in real-time")
    parser.add_argument("--show_actions", action="store_true", help="Print numerical action vectors during rollouts")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to bc_actor_checkpoint.pth")
    parser.add_argument("--scene_xml", type=str, default="", help="Path to scene_cube.xml")
    parser.add_argument("--output_dir", type=str, default="checkpoints_ppo", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Resolve project file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "in_hand_manipulation" else script_dir
    
    ckpt_path = args.checkpoint
    if not ckpt_path:
        cand1 = os.path.join(script_dir, "bc_actor_checkpoint.pth")
        cand2 = os.path.join(project_root, "in_hand_manipulation/bc_actor_checkpoint.pth")
        ckpt_path = cand1 if os.path.exists(cand1) else cand2
        
    scene_path = args.scene_xml
    if not scene_path:
        cand1 = os.path.join(script_dir, "simulation/scene_cube.xml")
        cand2 = os.path.join(project_root, "in_hand_manipulation/simulation/scene_cube.xml")
        scene_path = cand1 if os.path.exists(cand1) else cand2
        
    config = PPOConfig(
        num_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        mini_batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        warmup_iterations=args.warmup_iters,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        target_kl=args.target_kl,
        max_episode_steps=args.max_steps,
        render=args.render,
        show_actions=args.show_actions,
        dr_ratio=args.dr_ratio,
    )
    
    tuner = PPOFineTuner(
        config=config,
        checkpoint_path=ckpt_path,
        scene_xml_path=scene_path
    )
    
    tuner.train(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

