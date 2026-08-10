import os
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time

class TesolloInHandEnv(gym.Env):
    """
    Custom Gymnasium Environment for In-Hand Manipulation with TESOLLO DG-3F.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path="scene.xml", render_mode=None):
        super().__init__()
        
        # Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # 12 Actuators (Target Joint Positions)
        self.num_actuators = self.model.nu
        
        # Action Space: 12 continuous normalized joint position delta targets [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32
        )

        # Observation Space: Joint Positions (12) + Joint Velocities (12) + Object Position (3) + Object Orientation Quaternion (4) + Goal Quaternion (4)
        obs_dim = self.num_actuators * 2 + 3 + 4 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Default Goal: Target orientation quaternion
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_obs(self):
        # 1. Joint positions and velocities
        qpos = self.data.qpos[:self.num_actuators].copy()
        qvel = self.data.qvel[:self.num_actuators].copy()
        
        # 2. Object pose (assumes target_object is the freejoint body)
        obj_pos = self.data.body("target_object").xpos.copy()
        obj_quat = self.data.body("target_object").xquat.copy()

        return np.concatenate([qpos, qvel, obj_pos, obj_quat, self.goal_quat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset physics state
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial forward step to populate kinematic values
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        # Scale normalized action [-1, 1] to physical joint limits
        ctrl_ranges = self.model.actuator_ctrlrange
        scaled_action = ctrl_ranges[:, 0] + (action + 1.0) * 0.5 * (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])
        
        # Apply target joint positions
        self.data.ctrl[:] = scaled_action

        # Step physics forward (Substepping for 50Hz control frequency)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Compute Observation, Reward, and Termination
        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_status()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_reward_and_status(self):
        obj_pos = self.data.body("target_object").xpos
        obj_quat = self.data.body("target_object").xquat

        # Distance penalty (keep object in palm)
        palm_pos = np.array([0.0, 0.0, 0.15])
        dist_to_palm = np.linalg.norm(obj_pos - palm_pos)

        # Orientation error relative to goal quaternion
        quat_diff = np.abs(np.dot(obj_quat, self.goal_quat))
        quat_diff = np.clip(quat_diff, -1.0, 1.0)
        orient_error = 2.0 * np.arccos(quat_diff)

        # Dense Reward
        reward = -1.0 * orient_error - 2.0 * dist_to_palm

        # Terminate episode if object falls out of palm envelope
        terminated = dist_to_palm > 0.12

        return reward, terminated

    def render(self):
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print("Failed to launch MuJoCo viewer:", e)
                self.viewer = None
                return

        try:
            self.viewer.sync()
        except Exception as e:
            print("Viewer sync failed:", e)
            self.viewer = None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


if __name__ == "__main__":
    env = TesolloInHandEnv(model_path="/mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/in_hand_manipulation/simulation/scene.xml")
    obs, info = env.reset()

    print("Launching viewer to watch random actions...")
    try:
        for _ in range(500):  # Run for 500 steps
            random_action = env.action_space.sample()  # Brainless random twitching
            obs, reward, terminated, truncated, info = env.step(random_action)

            env.render()
            time.sleep(0.02)

            if terminated:
                print("Object dropped! Resetting...")
                env.reset()
    finally:
        env.close()
