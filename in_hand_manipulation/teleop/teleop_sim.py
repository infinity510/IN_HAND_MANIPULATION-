import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import cv2
from pynput import keyboard
from dex_retargeting.retargeting_config import RetargetingConfig
from webcam_aruco_tracker import WebcamArucoTracker
import os

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "../simulation/gripper_only.xml")
RETARGET_YML = os.path.join(os.path.dirname(__file__), "delto_3f_retarget.yml")

CONTROL_HZ = 20
SIM_HZ = 500
SUBSTEPS = SIM_HZ // CONTROL_HZ
MAX_JOINT_VEL = 0.5 

class TeleopSystem:
    def __init__(self):
        # Using the new USB Webcam tracker. 
        # Note: If your external USB webcam is not found, you may need to change camera_index=1 or 2
        self.tracker = WebcamArucoTracker(marker_length=0.015, camera_index=1)
        self.tracker.start()
        
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        config = RetargetingConfig.load_from_file(RETARGET_YML)
        self.retargeter = config.build()
        
        self.clutch_active = False
        self.absolute_mode = False
        self.calibrated_pinch_distances = None
        self.calibrated_pinch_angles = None
        self.local_centroid = None
        self.trigger_reset = False
        self.human_anchor = None
        self.robot_anchor = None
        
        self.joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        
        self.actuator_ids = []
        for name in self.joint_names:
            act_name = name.replace("gripper_", "act_").replace("_joint", "")
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_ids.append(act_id)

    def get_robot_tip_positions(self):
        # Use Dex-Retargeting's internal FK to prevent any anchor mismatch jumps!
        robot = self.retargeter.optimizer.robot
        robot.compute_forward_kinematics(self.last_qpos)
        positions = []
        for idx in self.retargeter.optimizer.target_link_indices:
            positions.append(robot.get_link_pose(idx)[:3, 3].copy())
        return np.array(positions)

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.clutch_active = True
        try:
            if key.char == 'o':
                self.absolute_mode = not getattr(self, 'absolute_mode', False)
                logging.info(f"Absolute Mode (O-key): {self.absolute_mode}")
            elif key.char == 'r':
                self.trigger_reset = True
        except: pass

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.clutch_active = False
            self.human_anchor = None
            self.robot_anchor = None

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        mujoco.mj_resetData(self.model, self.data)
        
        # INITIALIZE TO A PINCHED 120-DEGREE TRIPOD GRASP
        initial_pose = [
            ("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047),
            ("gripper_f1m3_joint", 1.0), ("gripper_f1m4_joint", 1.0),
            ("gripper_f2m3_joint", 1.0), ("gripper_f2m4_joint", 1.0),
            ("gripper_f3m3_joint", 1.0), ("gripper_f3m4_joint", 1.0),
        ]
        for name, val in initial_pose:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1: self.data.qpos[self.model.jnt_qposadr[jid]] = val
            except: pass
            
        mujoco.mj_forward(self.model, self.data)
        qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.last_qpos = np.array([self.data.qpos[idx] for idx in qpos_indices])
        
        for i, aid in enumerate(self.actuator_ids):
            if aid != -1: self.data.ctrl[aid] = self.last_qpos[i]

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                logging.info("=======================================")
                logging.info(" Teleop Running! (20 Hz)               ")
                logging.info(" Press 'o' to toggle ABSOLUTE MODE.    ")
                logging.info(" Hold SPACEBAR for relative mode.      ")
                logging.info("=======================================")
                
                while viewer.is_running():
                    step_start = time.time()
                    
                    if getattr(self, 'trigger_reset', False):
                        self.trigger_reset = False
                        
                        self.last_qpos = np.zeros_like(self.last_qpos)
                        reset_pose = [
                            ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047),
                            ("gripper_f1m3_joint", 1.0), ("gripper_f1m4_joint", 1.0),
                            ("gripper_f2m3_joint", 1.0), ("gripper_f2m4_joint", 1.0),
                            ("gripper_f3m3_joint", 1.0), ("gripper_f3m4_joint", 1.0),
                        ]
                        for name, val in reset_pose:
                            try:
                                idx = self.joint_names.index(name)
                                self.last_qpos[idx] = val
                            except: pass
                            
                        for i, aid in enumerate(self.actuator_ids):
                            if aid != -1: self.data.ctrl[aid] = self.last_qpos[i]
                            
                        for i, name in enumerate(self.joint_names):
                            try:
                                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                if jid != -1:
                                    self.data.qpos[self.model.jnt_qposadr[jid]] = self.last_qpos[i]
                                    self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
                            except: pass
                            
                        mujoco.mj_forward(self.model, self.data)
                        if hasattr(self.retargeter, 'set_qpos'):
                            self.retargeter.set_qpos(self.last_qpos)
                            
                        self.clutch_active = False
                        self.human_anchor = None
                        self.robot_anchor = None
                        logging.info("Reset to initial starting pose.")
                        
                    tracking_out, frame = self.tracker.step()
                    
                    if frame is not None:
                        mode_text = "MODE: ABSOLUTE (O)" if self.absolute_mode else ("MODE: RELATIVE (SPACE)" if self.clutch_active else "MODE: PAUSED")
                        color = (0, 255, 255) if self.absolute_mode else ((0, 255, 0) if self.clutch_active else (0, 0, 255))
                        cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.imshow("ArUco Teleop Feed", frame)
                        cv2.waitKey(1)
                    
                    if tracking_out is not None:
                        valid_tracking = True
                        human_pos = []
                        for m_id in [0, 1, 2]:
                            pos = tracking_out.get(m_id)
                            if pos is None:
                                valid_tracking = False
                                break
                            human_pos.append(pos)
                        
                        target_robot_pos = None
                        direct_qpos = None
                        target_qpos = None
                        
                        wrist_pos = tracking_out.get(3)
                        if wrist_pos is None:
                            valid_tracking = False
                        
                                                if self.absolute_mode or self.clutch_active:
                            human_pos = np.array(human_pos)
                            centroid = wrist_pos
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
                                
                            if self.local_centroid is None:
                                self.local_centroid = np.mean(aligned, axis=0)
                                vecs_curl = aligned[:, :2] - self.local_centroid[:2]
                                self.calibrated_pinch_distances = [np.linalg.norm(v) for v in vecs_curl]
                                self.calibrated_pinch_angles = [np.arctan2(aligned[i, 1], aligned[i, 0]) for i in range(3)]
                                
                            if self.absolute_mode:
                                curls = []
                                sways = []
                                for i in range(3):
                                    vec_for_curl = aligned[i, :2] - self.local_centroid[:2]
                                    r = np.linalg.norm(vec_for_curl)
                                    
                                    angle = np.arctan2(aligned[i, 1], aligned[i, 0])
                                    
                                    curl_i = 0.8 - (r - self.calibrated_pinch_distances[i]) * 25.0
                                    curl_i = np.clip(curl_i, -0.2, 2.0)
                                    curls.append(curl_i)
                                    
                                    delta_angle = angle - self.calibrated_pinch_angles[i]
                                    delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
                                    sway_i = np.clip(delta_angle * 1.5, -0.6, 0.6)
                                    sways.append(sway_i)
                                    
                                direct_qpos = self.last_qpos.copy()
                                for j_name, base_val in [("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047)]:
                                    try:
                                        idx = self.joint_names.index(j_name)
                                        direct_qpos[idx] = base_val
                                    except: pass
                                    
                                for i, f_idx in enumerate([1, 2, 3]):
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                        direct_qpos[idx] = sways[i]
                                    except: pass
                                    for m_idx in [3, 4]:
                                        try:
                                            idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                            direct_qpos[idx] = curls[i]
                                        except: pass
                                        
                                target_qpos = direct_qpos
                                
                            elif self.clutch_active:
                                finger_angles = []
                                finger_spreads = []
                                for i in range(3):
                                    vec_for_curl = aligned[i, :2] - self.local_centroid[:2]
                                    finger_spreads.append(np.linalg.norm(vec_for_curl))
                                    finger_angles.append(np.arctan2(aligned[i, 1], aligned[i, 0]))
                                    
                                if self.human_anchor is None:
                                    self.human_anchor = True
                                    self.init_theta = theta_hand
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
                                    
                                sway_scales = [0.5, 3.5, 3.5]  # Thumb (f1) scaled down, Index/Middle (f2,f3) scaled up
                                for i, f_idx in enumerate([1, 2, 3]):
                                    angle_delta = finger_angles[i] - self.calibrated_pinch_angles[i]
                                    angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
                                    
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                        target_qpos[idx] = np.clip(self.init_robot_qpos[idx] + angle_delta * sway_scales[i], -0.6, 0.6)
                                    except: pass
                                    
                                    r = finger_spreads[i]
                                    curl_i = 0.8 - (r - self.calibrated_pinch_distances[i]) * 25.0
                                    curl_i = np.clip(curl_i, -0.2, 2.2)
                                    
                                    for m_idx in [3, 4]:
                                        try:
                                            idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                            target_qpos[idx] = curl_i
                                        except: pass
                            
                        if target_qpos is not None:
                            delta_q = target_qpos - self.last_qpos
                            delta_q = np.clip(delta_q, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                            safe_qpos = self.last_qpos + delta_q
                            
                            for i, aid in enumerate(self.actuator_ids):
                                if aid != -1:
                                    self.data.ctrl[aid] = safe_qpos[i]
                                    
                            self.last_qpos = safe_qpos
                            
                    for _ in range(SUBSTEPS):
                        mujoco.mj_step(self.model, self.data)
                        
                    viewer.sync()
                    
                    elapsed = time.time() - step_start
                    time_to_wait = (1.0 / CONTROL_HZ) - elapsed
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
                        
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.tracker.stop()
            listener.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = TeleopSystem()
    system.run()
