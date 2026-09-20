import re

with open('collect_data.py', 'r') as f:
    content = f.read()

# Replace dict initialization
content = content.replace(
    "self.episode_data = {'qpos': [], 'qvel': [], 'action': []}",
    "self.episode_data = {'robot_qpos': [], 'robot_qvel': [], 'object_pos': [], 'object_quat': [], 'action': []}"
)

# Replace save_episode logic
save_orig = """    def save_episode(self):
        if len(self.episode_data['qpos']) < 10:
            logging.warning("Episode too short, discarding.")
        else:
            filename = f"data/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
            with h5py.File(filename, 'w') as f:
                f.create_dataset('qpos', data=np.array(self.episode_data['qpos']))
                f.create_dataset('qvel', data=np.array(self.episode_data['qvel']))
                f.create_dataset('action', data=np.array(self.episode_data['action']))
                f.attrs['object_id'] = self.active_object if self.active_object is not None else 0
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['qpos'])} steps. (Object: {self.active_object})")
            self.episode_counter += 1"""

save_repl = """    def save_episode(self):
        if len(self.episode_data['robot_qpos']) < 10:
            logging.warning("Episode too short, discarding.")
        else:
            filename = f"data/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
            with h5py.File(filename, 'w') as f:
                f.create_dataset('robot_qpos', data=np.array(self.episode_data['robot_qpos']))
                f.create_dataset('robot_qvel', data=np.array(self.episode_data['robot_qvel']))
                f.create_dataset('object_pos', data=np.array(self.episode_data['object_pos']))
                f.create_dataset('object_quat', data=np.array(self.episode_data['object_quat'])) # MuJoCo format: [w, x, y, z]
                f.create_dataset('action', data=np.array(self.episode_data['action']))
                f.attrs['object_id'] = self.active_object if self.active_object is not None else 0
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['robot_qpos'])} steps. (Object: {self.active_object})")
            self.episode_counter += 1"""
content = content.replace(save_orig, save_repl)

# Replace recording logic
rec_orig = """                    if not hasattr(self, 'last_record_time') or (current_time - self.last_record_time) >= 0.02:
                        self.episode_data['qpos'].append(self.data.qpos.copy())
                        self.episode_data['qvel'].append(self.data.qvel.copy())
                        self.episode_data['action'].append(self.data.ctrl.copy())
                        self.last_record_time = current_time"""

rec_repl = """                    if not hasattr(self, 'last_record_time') or (current_time - self.last_record_time) >= 0.02:
                        # Extract explicit features for Isaac Sim portability
                        r_qpos = [self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] for j in self.joint_names]
                        r_qvel = [self.data.qvel[self.model.jnt_dofadr[self.model.joint(j).id]] for j in self.joint_names]
                        
                        obj_pos = [0.0, 0.0, 0.0]
                        obj_quat = [1.0, 0.0, 0.0, 0.0]
                        if self.active_object is not None:
                            obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{self.active_object}_joint").id]
                            obj_pos = self.data.qpos[obj_idx:obj_idx+3].copy()
                            obj_quat = self.data.qpos[obj_idx+3:obj_idx+7].copy()
                            
                        self.episode_data['robot_qpos'].append(r_qpos)
                        self.episode_data['robot_qvel'].append(r_qvel)
                        self.episode_data['object_pos'].append(obj_pos)
                        self.episode_data['object_quat'].append(obj_quat)
                        self.episode_data['action'].append(self.data.ctrl.copy())
                        self.last_record_time = current_time"""
content = content.replace(rec_orig, rec_repl)

with open('collect_data.py', 'w') as f:
    f.write(content)

