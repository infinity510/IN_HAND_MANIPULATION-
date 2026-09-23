import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import glob
import h5py
from pynput import keyboard

# Find data dir
data_dir = "data_processed"
if not os.path.exists(data_dir):
    data_dir = "../data_processed"
    if not os.path.exists(data_dir):
        data_dir = "../../data_processed"
        if not os.path.exists(data_dir):
            data_dir = "/mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/data_processed"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "../simulation/scene_cube.xml")

class ReplaySystem:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt) if self.model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
        
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")), key=natural_sort_key)
        if not self.files:
            print(f"No HDF5 files found in {data_dir}")
            exit(1)
            
        self.current_ep_idx = 0
        self.is_paused = False
        self.step_idx = 0
        self.request_next = False
        self.request_prev = False
        
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.is_paused = not self.is_paused
                state = "PAUSED" if self.is_paused else "PLAYING"
                print(f"[{state}]")
            elif key == keyboard.Key.right:
                self.request_next = True
            elif key == keyboard.Key.left:
                self.request_prev = True
        except AttributeError:
            pass

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Enable contact point rendering
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            while viewer.is_running():
                if self.current_ep_idx >= len(self.files):
                    self.current_ep_idx = 0
                if self.current_ep_idx < 0:
                    self.current_ep_idx = len(self.files) - 1
                
                filepath = self.files[self.current_ep_idx]
                
                with h5py.File(filepath, 'r') as f:
                    robot_qpos = f['robot_qpos'][:]
                    object_pos = f['object_pos'][:]
                    object_quat = f['object_quat'][:]
                    obj_id = f.attrs.get('object_id', 0)
                    
                    num_steps = len(robot_qpos)
                    self.step_idx = 0
                    
                    print(f"Replaying {os.path.basename(filepath)} ({num_steps} steps) - [Space] Pause/Play  [Left] Prev  [Right] Next")
                    
                    while self.step_idx < num_steps and viewer.is_running():
                        step_start = time.time()
                        
                        if self.request_next:
                            self.current_ep_idx += 1
                            self.request_next = False
                            break
                        if self.request_prev:
                            self.current_ep_idx -= 1
                            self.request_prev = False
                            break
                            
                        if not self.is_paused:
                            # Move all objects away
                            for i in range(1, 4):
                                try:
                                    idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
                                    if i == obj_id:
                                        self.data.qpos[idx:idx+3] = object_pos[self.step_idx]
                                        self.data.qpos[idx+3:idx+7] = object_quat[self.step_idx]
                                    else:
                                        self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                                except Exception:
                                    pass
                            
                            # Set robot joints
                            for j, j_name in enumerate(self.joint_names):
                                try:
                                    qpos_idx = self.model.jnt_qposadr[self.model.joint(j_name).id]
                                    self.data.qpos[qpos_idx] = robot_qpos[self.step_idx][j]
                                except Exception:
                                    pass
                                    
                            mujoco.mj_forward(self.model, self.data)
                            viewer.sync()
                            self.step_idx += 1
                            
                        # Keep playback roughly real-time (data was recorded at ~50 Hz, loop time ~0.02)
                        elapsed = time.time() - step_start
                        time_until_next = 0.02 - elapsed
                        if time_until_next > 0:
                            time.sleep(time_until_next)
                    
                    # If episode naturally ended, go to next
                    if self.step_idx >= num_steps and viewer.is_running() and not self.request_prev and not self.request_next:
                        self.current_ep_idx += 1
                        time.sleep(0.5)

if __name__ == "__main__":
    system = ReplaySystem()
    system.run()
