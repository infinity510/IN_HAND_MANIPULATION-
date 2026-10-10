import os
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_bc import ActorNetwork

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "simulation/scene_cube.xml")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "bc_actor_checkpoint.pth")

class BCEvaluator:
    def __init__(self):
        print("Initializing MuJoCo...")
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        print("Loading BC Actor Network...")
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please run train_bc.py first!")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']
        hp = checkpoint['hyperparameters']
        
        self.actor = ActorNetwork(obs_dim=hp['obs_dim'], act_dim=hp['act_dim'], hidden_dim=hp['hidden_dims'][0])
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        self.actor.to(self.device)
        self.actor.eval()
        
        # State variables
        self.is_running = True
        self.active_object = 1 # Default to cylinder
        self.spawn_requested = 1
        self.randomize_target_requested = False
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0]) # [w, x, y, z]
        
        self.joint_names = self.get_gripper_joint_names()
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name.replace('gripper_', '').replace('_joint', '')}") for name in self.joint_names]
        
    def get_gripper_joint_names(self):
        names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name.startswith("gripper_") and "joint" in name:
                names.append(name)
        return names

    def on_press(self, key):
        try:
            if key.char in ['1', '2', '3']:
                self.spawn_requested = int(key.char)
            elif key.char == 'r':
                self.randomize_target_requested = True
            elif key.char == 'q':
                self.is_running = False
        except AttributeError:
            pass
            
    def get_physical_params(self, obj_id):
        # [radius/width, height, mass, friction]
        if obj_id == 1:
            return [0.045, 0.07, 1.0, 3.0]
        elif obj_id == 2:
            return [0.045, 0.045, 1.0, 3.0]
        elif obj_id == 3:
            return [0.055, 0.07, 1.0, 3.0]
        return [0.045, 0.07, 1.0, 3.0]

    def run(self):
        print("\n" + "="*50)
        print("LIVE BC EVALUATION STARTED")
        print("The script will automatically switch objects and target orientations every 5 seconds.")
        print("Close the MuJoCo window to stop.")
        print("="*50 + "\n")
        
        mujoco.mj_resetData(self.model, self.data)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            last_switch_time = time.time()
            self.active_object = 1
            self.spawn_requested = self.active_object
            self.randomize_target_requested = True
            
            while viewer.is_running() and self.is_running:
                step_start = time.time()
                
                # Automatically cycle objects and targets every 5 seconds
                if time.time() - last_switch_time > 5.0:
                    self.active_object += 1
                    if self.active_object > 3:
                        self.active_object = 1
                    self.spawn_requested = self.active_object
                    self.randomize_target_requested = True
                    last_switch_time = time.time()
                
                # Handle Object Spawning
                if self.spawn_requested is not None:
                    obj_id = self.spawn_requested
                    self.spawn_requested = None
                    print(f"\n---> Spawning object {obj_id}...")
                    
                    for i in range(1, 4):
                        try:
                            idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
                            vel_idx = self.model.jnt_dofadr[self.model.joint(f"obj_{i}_joint").id]
                            if i == obj_id:
                                # Spawn at origin
                                self.data.qpos[idx:idx+7] = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                            else:
                                # Move others far away
                                self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                            self.data.qvel[vel_idx:vel_idx+6] = 0.0
                        except Exception:
                            pass
                            
                # Handle Target Randomization
                if self.randomize_target_requested:
                    self.randomize_target_requested = False
                    q = R.random().as_quat() # [x, y, z, w]
                    self.target_quat = np.array([q[3], q[0], q[1], q[2]]) # MuJoCo format: [w, x, y, z]
                    print(f"---> New Target Orientation: {np.round(self.target_quat, 2)}")
                
                # 1. Gather Observation
                r_qpos = [self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] for j in self.joint_names]
                r_qvel = [self.data.qvel[self.model.jnt_dofadr[self.model.joint(j).id]] for j in self.joint_names]
                
                obj_pos = [0.0, 0.0, 0.0]
                obj_quat = [1.0, 0.0, 0.0, 0.0]
                try:
                    obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{self.active_object}_joint").id]
                    obj_pos = self.data.qpos[obj_idx:obj_idx+3].copy()
                    obj_quat = self.data.qpos[obj_idx+3:obj_idx+7].copy()
                except Exception:
                    pass
                    
                phys_params = self.get_physical_params(self.active_object)
                
                # Build 39-dim vector
                obs_raw = np.concatenate([r_qpos, r_qvel, obj_pos, obj_quat, self.target_quat, phys_params]).astype(np.float32)
                
                # Normalize
                obs_norm = (obs_raw - self.obs_mean) / self.obs_std
                
                # 2. Query Network for Action
                obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
                
                # 3. Apply Action (with clipping safety limit for velocities)
                current_qpos = np.array(r_qpos)
                delta_q = action - current_qpos
                delta_q = np.clip(delta_q, -0.15, 0.15) # Speed limit to prevent breaking physics
                safe_action = current_qpos + delta_q
                
                for i, aid in enumerate(self.actuator_ids):
                    if aid != -1:
                        self.data.ctrl[aid] = safe_action[i]
                        
                # 4. Step Simulation (Multiple substeps for stability)
                for _ in range(10): # 10 substeps per control loop
                    mujoco.mj_step(self.model, self.data)
                    
                viewer.sync()
                
                # Sleep to maintain approx realtime (assuming 0.002s timestep * 10 = 0.02s per loop)
                time_until_next_step = 0.02 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
if __name__ == "__main__":
    evaluator = BCEvaluator()
    evaluator.run()

