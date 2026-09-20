import re

with open('collect_data.py', 'r') as f:
    content = f.read()

# 1. Add logic in run() for spawning
run_orig = """                if self.reset_requested:
                    self.reset_to_start()
                    self.reset_requested = False
                    
                target_qpos = None"""

run_repl = """                if self.reset_requested:
                    self.reset_to_start()
                    self.reset_requested = False
                    
                if self.spawn_object_requested is not None:
                    obj_id = self.spawn_object_requested
                    self.spawn_object_requested = None
                    self.active_object = obj_id
                    logging.info(f"Spawning object {obj_id} into workspace...")
                    for i in range(1, 6):
                        try:
                            idx = self.model.jnt_qposadr[self.model.joint(f"obj_{i}_joint").id]
                            vel_idx = self.model.jnt_dofadr[self.model.joint(f"obj_{i}_joint").id]
                            if i == obj_id:
                                self.data.qpos[idx:idx+7] = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                            else:
                                self.data.qpos[idx:idx+7] = [10.0 + i, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
                            self.data.qvel[vel_idx:vel_idx+6] = 0.0
                        except Exception as e:
                            pass
                    
                target_qpos = None"""
content = content.replace(run_orig, run_repl)

# 2. Fix the Cube Pose extraction logic which was looking for "cube_joint"
hud_orig = """                # Extract Cube Pose for Display
                try:
                    cube_idx = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
                    quat = self.data.qpos[cube_idx+3:cube_idx+7]
                    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)
                    contact_msg = "TOUCHING" if self.data.ncon > 0 else "NO CONTACT"
                    overlay_text = f"Cube Roll/Pitch/Yaw:\n{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}\nStatus: {contact_msg}"
                except Exception as e:
                    overlay_text = f"Cube Error:\n{e}\n\"""
"""

hud_repl = """                # Extract Active Object Pose for Display
                try:
                    if self.active_object is not None:
                        obj_idx = self.model.jnt_qposadr[self.model.joint(f"obj_{self.active_object}_joint").id]
                        quat = self.data.qpos[obj_idx+3:obj_idx+7]
                        euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)
                        contact_msg = "TOUCHING" if self.data.ncon > 0 else "NO CONTACT"
                        obj_names = {1: 'Cylinder', 2: 'Capsule (Cone)', 3: 'Cube', 4: 'Cuboid', 5: 'Sphere'}
                        obj_name = obj_names.get(self.active_object, 'Unknown')
                        overlay_text = f"{obj_name} Roll/Pitch/Yaw:\n{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}\nStatus: {contact_msg}"
                    else:
                        overlay_text = "No Object Spawned\nPress 1-5 to spawn\nStatus: WAITING"
                except Exception as e:
                    overlay_text = f"Object Error:\n{e}\n\"""
"""
content = content.replace(hud_orig, hud_repl)

with open('collect_data.py', 'w') as f:
    f.write(content)
