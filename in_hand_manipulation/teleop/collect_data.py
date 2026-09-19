import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
from pynput import keyboard
from webcam_aruco_tracker import WebcamArucoTracker
import os
import h5py
from datetime import datetime
from scipy.spatial.transform import Rotation as R

logging = __import__("logging")
logging.basicConfig(level=logging.INFO)

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "../simulation/scene_cube.xml")
POSITION_SCALING = 1.5

R_CAM2ROB = np.array([
    [ 0, -1,  0],  
    [-1,  0,  0],  
    [ 0,  0, -1]   
])

class TeleopSystem:
    def __init__(self):
        self.tracker = WebcamArucoTracker(marker_length=0.015, camera_index=4)
        self.tracker.start()
        
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt) if self.model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        
        self.clutch_active = False
        self.absolute_mode = False
        self.reset_requested = False
        
        self.last_qpos = np.zeros(12)
        self.human_anchor = None
        self.init_robot_qpos = None
        
        # Data Collection
        self.is_recording = False
        self.episode_data = {'qpos': [], 'qvel': [], 'action': []}
        self.episode_counter = 1
        
        # Ensure data dir exists
        os.makedirs("data", exist_ok=True)
        
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        self.reset_to_start()
        mujoco.mj_step(self.model, self.data)

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if not self.clutch_active:
                    self.clutch_active = True
                    logging.info("Clutch ENGAGED")
            elif key.char == 'c':
                if not self.is_recording:
                    self.is_recording = True
                    logging.info("RECORDING STARTED")
                else:
                    if self.is_recording:
                        self.save_episode()
                    self.is_recording = False
                    logging.info("RECORDING STOPPED & SAVED")
            elif key.char == 'o':
                self.absolute_mode = True
                logging.info("Absolute Mode ENABLED")
            elif key.char == 'r':
                self.reset_requested = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.space:
                self.clutch_active = False
                self.human_anchor = None
                logging.info("Clutch RELEASED")
            elif key.char == 'o':
                self.absolute_mode = False
                logging.info("Absolute Mode DISABLED")
        except AttributeError:
            pass

    def save_episode(self):
        if len(self.episode_data['qpos']) < 10:
            logging.warning("Episode too short, discarding.")
        else:
            filename = f"data/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
            with h5py.File(filename, 'w') as f:
                f.create_dataset('qpos', data=np.array(self.episode_data['qpos']))
                f.create_dataset('qvel', data=np.array(self.episode_data['qvel']))
                f.create_dataset('action', data=np.array(self.episode_data['action']))
            logging.info(f"Saved episode {self.episode_counter} to {filename} with {len(self.episode_data['qpos'])} steps.")
            self.episode_counter += 1
            
        self.episode_data = {'qpos': [], 'qvel': [], 'action': []}

    def reset_to_start(self):
        logging.info("Resetting robot to initial pinch tripod...")
        self.clutch_active = False
        self.human_anchor = None
        self.is_recording = False
        self.episode_data = {'qpos': [], 'qvel': [], 'action': []}
        
        initial_pose = [
            ("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047),
            ("gripper_f1m3_joint", 1.0), ("gripper_f1m4_joint", 1.0),
            ("gripper_f2m3_joint", 1.0), ("gripper_f2m4_joint", 1.0),
            ("gripper_f3m3_joint", 1.0), ("gripper_f3m4_joint", 1.0),
        ]
        
        for i in range(self.model.nu):
            self.data.ctrl[i] = 0.0
            
        for name, val in initial_pose:
            try:
                idx = self.joint_names.index(name)
                self.data.ctrl[idx] = val
                # Freejoints offset the qpos array by 7!
                qpos_idx = self.model.jnt_qposadr[self.model.joint(name).id]
                self.data.qpos[qpos_idx] = val
                self.last_qpos[idx] = val
            except ValueError:
                pass
                
        # Set cube to perfect grasp position
        try:
            cube_idx = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
            self.data.qpos[cube_idx:cube_idx+7] = [0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[:] = 0.0
        except Exception as e:
            pass

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # Enable contact point rendering
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            # Sync tracker thread
            time.sleep(1.0)
            
            while viewer.is_running():
                step_start = time.time()
                
                if self.reset_requested:
                    self.reset_to_start()
                    self.reset_requested = False
                    
                target_qpos = None
                
                tracking_out, color_image = self.tracker.step()
                if tracking_out is not None:
                    valid_tracking = True
                    human_pos = []
                    for m_id in [0, 1, 2]:
                        pos = tracking_out.get(m_id)
                        if pos is None:
                            valid_tracking = False
                            break
                        human_pos.append(pos)
                    
                    if valid_tracking:
                        if self.absolute_mode:
                            human_pos = np.array(human_pos)
                            centroid = np.mean(human_pos, axis=0)
                            centered = human_pos - centroid
                            
                            p0 = centered[0, :2]
                            p1 = centered[1, :2]
                            p2 = centered[2, :2]
                            v_human = p0 - (p1 + p2) / 2.0
                            v_len = np.linalg.norm(v_human) + 1e-6
                            v_human /= v_len
                            theta_hand = np.arctan2(v_human[1], v_human[0])
                            
                            c, s = np.cos(-theta_hand), np.sin(-theta_hand)
                            R_2d = np.array([[c, -s], [s, c]])
                            
                            aligned = np.zeros_like(centered)
                            for i in range(3):
                                aligned[i, :2] = R_2d @ centered[i, :2]
                                
                            curls = []
                            for i in range(3):
                                pt = aligned[i, :2]
                                r = np.linalg.norm(pt)
                                curl_i = 1.7 - (r - 0.02) * 25.0
                                curl_i = np.clip(curl_i, -0.2, 2.0)
                                curls.append(curl_i)
                                
                            direct_qpos = self.last_qpos.copy()
                            for j_name, base_val in [("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047)]:
                                try:
                                    idx = self.joint_names.index(j_name)
                                    direct_qpos[idx] = base_val
                                except: pass
                                
                            for i, f_idx in enumerate([1, 2, 3]):
                                try:
                                    idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                    direct_qpos[idx] = 0.0
                                except: pass
                                for m_idx in [3, 4]:
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                        direct_qpos[idx] = curls[i]
                                    except: pass
                                    
                            target_qpos = direct_qpos

                        elif self.clutch_active:
                            human_pos = np.array(human_pos)
                            current_centroid = np.mean(human_pos, axis=0)
                            centered = human_pos - current_centroid
                            
                            p0, p1, p2 = centered[0, :2], centered[1, :2], centered[2, :2]
                            v_human = p0 - (p1 + p2) / 2.0
                            theta_hand = np.arctan2(v_human[1], v_human[0])
                            
                            c, s = np.cos(-theta_hand), np.sin(-theta_hand)
                            R_2d = np.array([[c, -s], [s, c]])
                            aligned = np.zeros_like(centered)
                            for i in range(3):
                                aligned[i, :2] = R_2d @ centered[i, :2]
                                
                            finger_angles = []
                            finger_spreads = []
                            for i in range(3):
                                pt = aligned[i, :2]
                                finger_spreads.append(np.linalg.norm(pt))
                                finger_angles.append(np.arctan2(pt[1], pt[0]))
                                
                            if self.human_anchor is None:
                                self.human_anchor = True
                                self.init_theta = theta_hand
                                self.init_finger_angles = finger_angles
                                self.init_robot_qpos = self.last_qpos.copy()
                                
                            target_qpos = self.init_robot_qpos.copy()
                            
                            twist_delta = theta_hand - self.init_theta
                            twist_delta = (twist_delta + np.pi) % (2 * np.pi) - np.pi
                            twist_delta *= 1.2
                            
                            tripod_bases = [0.0, -1.047, 1.047]
                            for i, j_name in enumerate(["gripper_f1m1_joint", "gripper_f2m1_joint", "gripper_f3m1_joint"]):
                                try:
                                    idx = self.joint_names.index(j_name)
                                    target_qpos[idx] = np.clip(tripod_bases[i] + twist_delta, -1.9, 2.0)
                                except: pass
                                
                            for i, f_idx in enumerate([1, 2, 3]):
                                angle_delta = finger_angles[i] - self.init_finger_angles[i]
                                angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
                                
                                try:
                                    idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                    target_qpos[idx] = np.clip(self.init_robot_qpos[idx] + angle_delta * 1.5, -0.6, 0.6)
                                except: pass
                                
                                r = finger_spreads[i]
                                curl_i = 1.7 - (r - 0.02) * 25.0
                                curl_i = np.clip(curl_i, -0.2, 2.2)
                                
                                for m_idx in [3, 4]:
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                        target_qpos[idx] = curl_i
                                    except: pass
                
                if target_qpos is not None:
                    delta_q = target_qpos - self.last_qpos
                    target_qpos = self.last_qpos + np.clip(delta_q, -0.2, 0.2)
                    self.last_qpos = target_qpos
                    
                    for i, name in enumerate(self.joint_names):
                        try:
                            idx = self.actuator_names.index("act_" + name.replace("gripper_", "").replace("_joint", ""))
                            self.data.ctrl[idx] = target_qpos[i]
                        except ValueError:
                            pass
                            
                # Record Step if Recording (Downsampled to ~50 Hz for manageable dataset sizes)
                if self.is_recording:
                    current_time = time.time()
                    if not hasattr(self, 'last_record_time') or (current_time - self.last_record_time) >= 0.02:
                        self.episode_data['qpos'].append(self.data.qpos.copy())
                        self.episode_data['qvel'].append(self.data.qvel.copy())
                        self.episode_data['action'].append(self.data.ctrl.copy())
                        self.last_record_time = current_time
                    
                # Extract Cube Pose for Display
                try:
                    cube_idx = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
                    quat = self.data.qpos[cube_idx+3:cube_idx+7]
                    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)
                    contact_msg = "TOUCHING" if self.data.ncon > 0 else "NO CONTACT"
                    overlay_text = f"Cube Roll/Pitch/Yaw:\n{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}\nStatus: {contact_msg}"
                except Exception as e:
                    overlay_text = f"Cube Error:\n{e}\n"
                    
                # Highlight contacts in RED
                viewer.user_scn.ngeom = 0
                for c in range(self.data.ncon):
                    contact = self.data.contact[c]
                    
                    # Only draw if there's room in the pre-allocated user_scn array
                    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.array([1, 0, 0, 1])
                        )
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].size[:] = np.array([0.005, 0.005, 0.005])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].pos[:] = contact.pos
                        viewer.user_scn.ngeom += 1
                    
                if color_image is not None:
                    # Draw visual HUD on webcam
                    cv2.putText(color_image, f"REC: {self.is_recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.is_recording else (0,255,0), 2)
                    cv2.putText(color_image, overlay_text.split('\n')[0], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\n')[1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(color_image, overlay_text.split('\n')[2], (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if "TOUCHING" in overlay_text else (0,255,0), 2)
                    cv2.imshow("Aruco Tracking", color_image)
                    cv2.waitKey(1)
                    
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    system = TeleopSystem()
    system.run()
