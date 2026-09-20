import re

with open('collect_data.py', 'r') as f:
    content = f.read()

# 1. Update __init__
init_repl = """        self.init_robot_qpos = None
        
        self.active_object = None  # None means no object is spawned
"""
content = content.replace("        self.init_robot_qpos = None\n", init_repl)

# 2. Update on_press
on_press_orig = """            elif key.char == 'r':
                self.reset_requested = True
        except AttributeError:
            pass"""
on_press_repl = """            elif key.char == 'r':
                self.reset_requested = True
            elif key.char in ['1', '2', '3', '4', '5']:
                self.spawn_object_requested = int(key.char)
        except AttributeError:
            pass"""
content = content.replace(on_press_orig, on_press_repl)

# 2.5 add self.spawn_object_requested in __init__
init2_orig = """        self.absolute_mode = False
        self.reset_requested = False"""
init2_repl = """        self.absolute_mode = False
        self.reset_requested = False
        self.spawn_object_requested = None"""
content = content.replace(init2_orig, init2_repl)

# 3. Update save_episode
save_orig = """            with h5py.File(filename, 'w') as f:
                f.create_dataset('qpos', data=np.array(self.episode_data['qpos']))
                f.create_dataset('qvel', data=np.array(self.episode_data['qvel']))
                f.create_dataset('action', data=np.array(self.episode_data['action']))
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['qpos'])} steps.")"""
save_repl = """            with h5py.File(filename, 'w') as f:
                f.create_dataset('qpos', data=np.array(self.episode_data['qpos']))
                f.create_dataset('qvel', data=np.array(self.episode_data['qvel']))
                f.create_dataset('action', data=np.array(self.episode_data['action']))
                f.attrs['object_id'] = self.active_object if self.active_object is not None else 0
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['qpos'])} steps. (Object: {self.active_object})")"""
content = content.replace(save_orig, save_repl)

# 4. Update reset_to_start
reset_orig = """        # Set cube to perfect grasp position
        try:
            cube_idx = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
            self.data.qpos[cube_idx:cube_idx+7] = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[:] = 0.0
        except Exception as e:
            pass"""

reset_repl = """        # Move all objects away (clear workspace)
        self.active_object = None
        for i in range(1, 6):
            try:
                idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
                self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                
                # Reset velocities for the object
                vel_idx = self.model.jnt_dofadr[self.model.joint(f"obj_{i}_joint").id]
                self.data.qvel[vel_idx:vel_idx+6] = 0.0
            except Exception as e:
                pass"""
content = content.replace(reset_orig, reset_repl)


with open('collect_data.py', 'w') as f:
    f.write(content)

