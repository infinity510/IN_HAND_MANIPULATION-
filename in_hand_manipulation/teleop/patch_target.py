import re

with open('collect_data.py', 'r') as f:
    content = f.read()

# 1. Init target_quat
init_orig = """        # Data Collection
        self.is_recording = False
        self.episode_data = {'robot_qpos': [], 'robot_qvel': [], 'object_pos': [], 'object_quat': [], 'action': []}
        self.episode_counter = 1"""

init_repl = """        # Data Collection
        self.is_recording = False
        self.episode_data = {'robot_qpos': [], 'robot_qvel': [], 'object_pos': [], 'object_quat': [], 'action': []}
        self.episode_counter = 1
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]"""
content = content.replace(init_orig, init_repl)

# 2. Randomize target when recording starts
record_orig = """            elif key.char == 'c':
                if not self.is_recording:
                    self.is_recording = True
                    logging.info("RECORDING STARTED")
                else:
                    if self.is_recording:
                        self.save_episode()
                    self.is_recording = False
                    logging.info("RECORDING STOPPED & SAVED")"""

record_repl = """            elif key.char == 'c':
                if not self.is_recording:
                    self.is_recording = True
                    # Generate a random target orientation (Z-axis rotation mostly, with some tilt)
                    # For in-hand, full random might be too hard, but let's do a random rotation
                    rot = R.random().as_quat() # [x, y, z, w]
                    self.target_quat = np.array([rot[3], rot[0], rot[1], rot[2]]) # [w, x, y, z]
                    logging.info(f"RECORDING STARTED. Target Quat: {self.target_quat}")
                else:
                    if self.is_recording:
                        self.save_episode()
                    self.is_recording = False
                    logging.info("RECORDING STOPPED & SAVED")"""
content = content.replace(record_orig, record_repl)

# 3. Save target_quat as attribute
save_orig = """                f.create_dataset('action', data=np.array(self.episode_data['action']))
                f.attrs['object_id'] = self.active_object if self.active_object is not None else 0
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['robot_qpos'])} steps. (Object: {self.active_object})")"""

save_repl = """                f.create_dataset('action', data=np.array(self.episode_data['action']))
                f.attrs['object_id'] = self.active_object if self.active_object is not None else 0
                f.attrs['target_quat'] = self.target_quat
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['robot_qpos'])} steps. (Object: {self.active_object})")"""
content = content.replace(save_orig, save_repl)

# 4. Display in viewer and OpenCV
# We need to add the target drawing logic. We can just add it before viewer.sync()
draw_orig = """                if color_image is not None:
                    # Draw visual HUD on webcam
                    cv2.putText(color_image, f"REC: {self.is_recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.is_recording else (0,255,0), 2)
                    cv2.putText(color_image, overlay_text.split('\\n')[0], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\\n')[1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\\n')[2], (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if "TOUCHING" in overlay_text else (0,255,0), 2)"""

draw_repl = """                # Draw Target Coordinate Frame in Viewer
                if self.is_recording and viewer.user_scn.ngeom < viewer.user_scn.maxgeom - 3:
                    target_mat = R.from_quat([self.target_quat[1], self.target_quat[2], self.target_quat[3], self.target_quat[0]]).as_matrix()
                    target_pos = np.array([-0.1, 0.0, 0.15]) # Float to the side of the hand
                    colors = [np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])]
                    for i in range(3):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.zeros(9), colors[i]
                        )
                        # Create cylinder from center to axis tip
                        axis_vec = target_mat[:, i] * 0.05
                        center_pos = target_pos + axis_vec / 2.0
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].size[:] = np.array([0.003, 0.025, 0.0])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].pos[:] = center_pos
                        
                        # Rotate cylinder to match axis
                        z_axis = np.array([0, 0, 1])
                        v = np.cross(z_axis, target_mat[:, i])
                        c = np.dot(z_axis, target_mat[:, i])
                        if c > -0.999:
                            s = np.linalg.norm(v)
                            if s > 1e-5:
                                axis = v / s
                                angle = np.arccos(c)
                                q = R.from_rotvec(axis * angle).as_matrix().flatten()
                                viewer.user_scn.geoms[viewer.user_scn.ngeom].mat[:] = q
                        
                        viewer.user_scn.ngeom += 1

                if color_image is not None:
                    # Draw visual HUD on webcam
                    cv2.putText(color_image, f"REC: {self.is_recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.is_recording else (0,255,0), 2)
                    
                    if self.is_recording:
                        t_euler = R.from_quat([self.target_quat[1], self.target_quat[2], self.target_quat[3], self.target_quat[0]]).as_euler('xyz', degrees=True)
                        cv2.putText(color_image, f"TARGET: {t_euler[0]:.0f}, {t_euler[1]:.0f}, {t_euler[2]:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    
                    cv2.putText(color_image, overlay_text.split('\\n')[0], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\\n')[1], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\\n')[2], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if "TOUCHING" in overlay_text else (0,255,0), 2)"""

content = content.replace(draw_orig, draw_repl)

with open('collect_data.py', 'w') as f:
    f.write(content)

